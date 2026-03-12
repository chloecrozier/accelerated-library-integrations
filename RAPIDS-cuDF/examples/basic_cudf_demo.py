"""
Basic cuDF demo — GPU-accelerated DataFrame operations.

Generates synthetic sales data and runs common DataFrame operations
(filtering, grouping, joining, sorting, window functions) entirely
on the GPU using cuDF's pandas-like API.
"""

import cudf
import numpy as np

# ──────────────────────────────────────────────
# 1. Create a synthetic sales dataset on the GPU
# ──────────────────────────────────────────────
np.random.seed(42)
n_rows = 1_000_000

sales = cudf.DataFrame({
    "order_id": cudf.Series(range(n_rows)),
    "customer_id": cudf.Series(np.random.randint(1, 5_001, size=n_rows)),
    "product": cudf.Series(np.random.choice(
        ["Widget", "Gadget", "Doohickey", "Thingamajig", "Gizmo"],
        size=n_rows,
    )),
    "quantity": cudf.Series(np.random.randint(1, 20, size=n_rows)),
    "unit_price": cudf.Series(np.round(np.random.uniform(5.0, 500.0, size=n_rows), 2)),
    "region": cudf.Series(np.random.choice(
        ["North", "South", "East", "West"],
        size=n_rows,
    )),
})

sales["revenue"] = sales["quantity"] * sales["unit_price"]

print(f"Dataset: {len(sales):,} rows")
print(sales.head())
print()

# ──────────────────────────────────────────────
# 2. Filtering
# ──────────────────────────────────────────────
big_orders = sales[sales["revenue"] > 2000]
print(f"Orders over $2,000: {len(big_orders):,}")
print()

# ──────────────────────────────────────────────
# 3. Group-by aggregation
# ──────────────────────────────────────────────
by_product = (
    sales
    .groupby("product")
    .agg({"revenue": ["sum", "mean", "count"]})
)
by_product.columns = ["total_revenue", "avg_revenue", "order_count"]
by_product = by_product.sort_values("total_revenue", ascending=False)
print("Revenue by product:")
print(by_product)
print()

# ──────────────────────────────────────────────
# 4. Multi-key group-by
# ──────────────────────────────────────────────
by_region_product = (
    sales
    .groupby(["region", "product"])
    .agg({"revenue": "sum", "quantity": "sum"})
    .reset_index()
    .sort_values("revenue", ascending=False)
)
print("Top 5 region-product combos by revenue:")
print(by_region_product.head())
print()

# ──────────────────────────────────────────────
# 5. Create a second table and join
# ──────────────────────────────────────────────
customers = cudf.DataFrame({
    "customer_id": cudf.Series(range(1, 5_001)),
    "tier": cudf.Series(np.random.choice(
        ["Bronze", "Silver", "Gold", "Platinum"],
        size=5_000,
    )),
})

merged = sales.merge(customers, on="customer_id", how="left")
tier_summary = (
    merged
    .groupby("tier")
    .agg({"revenue": "sum", "order_id": "count"})
    .rename(columns={"order_id": "num_orders"})
    .sort_values("revenue", ascending=False)
)
print("Revenue by customer tier:")
print(tier_summary)
print()

# ──────────────────────────────────────────────
# 6. Sorting and top-N
# ──────────────────────────────────────────────
top_customers = (
    sales
    .groupby("customer_id")
    .agg({"revenue": "sum"})
    .sort_values("revenue", ascending=False)
    .head(10)
)
print("Top 10 customers by total revenue:")
print(top_customers)
print()

# ──────────────────────────────────────────────
# 7. Quick summary stats
# ──────────────────────────────────────────────
print("Revenue distribution:")
print(sales["revenue"].describe())
