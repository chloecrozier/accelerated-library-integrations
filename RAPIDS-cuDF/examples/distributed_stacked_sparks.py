"""
Distributed cuDF across stacked DGX Sparks with Dask.

Demonstrates GPU-accelerated DataFrame operations distributed across
two DGX Spark nodes connected via 200 Gb/s RoCE. Generates synthetic
data, partitions it across workers, and runs distributed group-by,
join, and shuffle operations.

Prerequisites
-------------
1. Two DGX Sparks connected via QSFP cable (see the stacked-sparks
   playbook: https://build.nvidia.com/spark/connect-two-sparks)
2. SSH configured between nodes (run discover-sparks on both)
3. Install on BOTH nodes:
       pip install dask[distributed] dask-cuda dask-cudf cudf-cu12 cupy-cuda12x

Usage
-----
Start the scheduler on Node 1 and a GPU worker on each node, then
run this script on Node 1:

    # Node 1 — scheduler + worker
    dask scheduler --interface enP2p1s0f1np1 &
    dask cuda worker <scheduler-ip>:8786 --interface enP2p1s0f1np1 --rmm-pool-size 80G &

    # Node 2 — worker only
    dask cuda worker <scheduler-ip>:8786 --interface enP2p1s0f1np1 --rmm-pool-size 80G &

    # Node 1 — run the script
    python distributed_stacked_sparks.py
"""

import argparse

import cupy as cp
import dask
import dask_cudf
from dask.distributed import Client, wait


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
SCHEDULER = "tcp://127.0.0.1:8786"
N_ROWS = 50_000_000       # 50M rows — scales well on 128 GB per node
N_PARTITIONS = 16          # spread across both workers
N_CUSTOMERS = 100_000


def make_sales(n_rows, n_partitions):
    """Generate a synthetic sales dask-cudf DataFrame."""
    import cudf

    parts = []
    rows_per = n_rows // n_partitions
    for _ in range(n_partitions):
        df = cudf.DataFrame({
            "customer_id": cp.random.randint(0, N_CUSTOMERS, size=rows_per),
            "product_id":  cp.random.randint(0, 500, size=rows_per),
            "quantity":    cp.random.randint(1, 20, size=rows_per),
            "unit_price":  cp.random.uniform(5.0, 500.0, size=rows_per).astype("float32"),
            "region":      cudf.Series(cp.random.randint(0, 4, size=rows_per)).astype("category"),
        })
        df["revenue"] = df["quantity"] * df["unit_price"]
        parts.append(df)

    return dask_cudf.from_cudf(cudf.concat(parts), npartitions=n_partitions)


def make_customers():
    """Generate a small customer-tier lookup table."""
    import cudf

    tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    return cudf.DataFrame({
        "customer_id": cp.arange(N_CUSTOMERS),
        "tier": cudf.Series((cp.random.randint(0, 4, size=N_CUSTOMERS)).tolist())
                    .map({0: tiers[0], 1: tiers[1], 2: tiers[2], 3: tiers[3]}),
    })


def run(scheduler: str):
    client = Client(scheduler)
    print(f"Connected to scheduler at {scheduler}")
    print(f"Workers: {len(client.scheduler_info()['workers'])}")
    print()

    # --- 1. Distributed data generation ---
    print("Generating 50M-row sales dataset across workers...")
    sales = make_sales(N_ROWS, N_PARTITIONS)
    sales = sales.persist()
    wait(sales)
    print(f"  Partitions: {sales.npartitions}")
    print(f"  Rows: {len(sales):,}")
    print()

    # --- 2. Distributed group-by aggregation ---
    #   This triggers a shuffle — data for each product_id must be
    #   co-located on the same worker before aggregation.
    print("Running distributed group-by (product_id)...")
    product_stats = (
        sales
        .groupby("product_id")
        .agg({"revenue": ["sum", "mean"], "quantity": "sum"})
        .persist()
    )
    wait(product_stats)
    product_stats.columns = ["total_revenue", "avg_revenue", "total_qty"]
    top5 = product_stats.nlargest(5, "total_revenue").compute()
    print("  Top 5 products by revenue:")
    print(top5.to_string(index=True))
    print()

    # --- 3. Distributed broadcast join ---
    #   The customer table is small enough to broadcast to every worker,
    #   avoiding a full shuffle. This is a coscheduling-friendly pattern:
    #   keep the large table partitioned in place, replicate the small one.
    print("Joining with customer tier table (broadcast)...")
    customers = make_customers()
    merged = sales.merge(
        dask_cudf.from_cudf(customers, npartitions=1),
        on="customer_id",
        how="left",
    ).persist()
    wait(merged)

    tier_summary = (
        merged
        .groupby("tier")
        .agg({"revenue": "sum", "customer_id": "count"})
        .compute()
        .rename(columns={"customer_id": "num_orders"})
        .sort_values("revenue", ascending=False)
    )
    print("  Revenue by customer tier:")
    print(tier_summary.to_string(index=True))
    print()

    # --- 4. Distributed shuffle join (large x large) ---
    #   A heavier operation: join sales against a per-customer summary.
    #   Both sides are large, so Dask must hash-partition and shuffle
    #   across workers — the 200 Gb/s RoCE link matters here.
    print("Computing per-customer lifetime value (shuffle join)...")
    cust_lifetime = (
        sales
        .groupby("customer_id")
        .agg({"revenue": "sum", "quantity": "sum"})
        .rename(columns={"revenue": "lifetime_revenue", "quantity": "lifetime_qty"})
        .persist()
    )
    wait(cust_lifetime)

    enriched = sales.merge(cust_lifetime, on="customer_id", how="left").persist()
    wait(enriched)
    print(f"  Enriched dataset rows: {len(enriched):,}")
    print(enriched.head())
    print()

    print("Done.")
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed cuDF demo across stacked DGX Sparks")
    parser.add_argument(
        "--scheduler", default=SCHEDULER,
        help="Dask scheduler address (default: tcp://127.0.0.1:8786)")
    args = parser.parse_args()
    run(args.scheduler)
