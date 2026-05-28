"""First Bowl of Soup: fraud ring detection with cuGraph.

This script creates a small synthetic financial-services graph where vertices
are accounts, devices, merchants, IP addresses, and phone numbers. Edges are
transactions or shared identifiers. cuGraph is then used to:

1. Find connected entity clusters.
2. Trace the neighborhood around a known-bad account with BFS.
3. Rank influential entities with PageRank.
4. Produce a small investigation queue that could feed a case-management tool.
"""

import argparse
import sys


try:
    import cudf
    import cugraph
    import pandas as pd
except ImportError as exc:
    sys.exit(
        "This example needs pandas, RAPIDS cuDF, and RAPIDS cuGraph on a "
        f"CUDA-capable Linux or WSL2 machine -> {exc}"
    )


def add_entity(entities, vertex, name, kind):
    entities[int(vertex)] = {"name": name, "kind": kind}


def add_edge(edges, src, dst, relation, weight):
    edges.append(
        {
            "src": int(src),
            "dst": int(dst),
            "relation": relation,
            "weight": float(weight),
        }
    )


def build_synthetic_fraud_data(benign_components):
    """Build a deterministic entity graph with one suspicious fraud ring."""
    entities = {}
    edges = []

    base_entities = {
        0: ("acct_known_bad", "account"),
        1: ("acct_mule_a", "account"),
        2: ("acct_mule_b", "account"),
        3: ("acct_mule_c", "account"),
        4: ("acct_victim", "account"),
        5: ("acct_customer_a", "account"),
        6: ("acct_customer_b", "account"),
        100: ("device_shared_emulator", "device"),
        101: ("device_family_phone", "device"),
        200: ("merchant_gift_cards", "merchant"),
        201: ("merchant_crypto_exchange", "merchant"),
        202: ("merchant_groceries", "merchant"),
        300: ("ip_proxy_datacenter", "ip"),
        301: ("ip_home_network", "ip"),
        400: ("phone_reused_voip", "phone"),
        401: ("phone_customer", "phone"),
    }

    for vertex, (name, kind) in base_entities.items():
        add_entity(entities, vertex, name, kind)

    fraud_edges = [
        (4, 0, "payment_to_known_bad", 3.0),
        (0, 1, "transfer", 2.5),
        (1, 2, "transfer", 2.0),
        (2, 3, "transfer", 2.0),
        (3, 1, "circular_transfer", 2.0),
        (1, 200, "purchase", 1.5),
        (2, 201, "purchase", 1.5),
        (0, 100, "login_device", 1.0),
        (1, 100, "login_device", 1.0),
        (2, 100, "login_device", 1.0),
        (3, 300, "login_ip", 1.0),
        (1, 400, "phone", 1.0),
        (2, 400, "phone", 1.0),
        (3, 400, "phone", 1.0),
    ]

    benign_edges = [
        (5, 101, "login_device", 1.0),
        (6, 101, "login_device", 1.0),
        (5, 202, "purchase", 1.0),
        (6, 202, "purchase", 1.0),
        (5, 301, "login_ip", 1.0),
        (6, 401, "phone", 1.0),
    ]

    for edge in fraud_edges + benign_edges:
        add_edge(edges, *edge)

    for i in range(benign_components):
        account = 1000 + i
        device = 2000 + i
        merchant = 3000 + i
        phone = 4000 + i

        add_entity(entities, account, f"acct_benign_{i:03d}", "account")
        add_entity(entities, device, f"device_benign_{i:03d}", "device")
        add_entity(entities, merchant, f"merchant_benign_{i:03d}", "merchant")
        add_entity(entities, phone, f"phone_benign_{i:03d}", "phone")

        add_edge(edges, account, device, "login_device", 0.5)
        add_edge(edges, account, merchant, "purchase", 0.5)
        add_edge(edges, account, phone, "phone", 0.5)

    edge_df = cudf.DataFrame(edges)
    return entities, edge_df


def component_labels(graph):
    try:
        return cugraph.weakly_connected_components(graph)
    except AttributeError:
        return cugraph.connected_components(graph, connection="weak")


def require_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    raise RuntimeError(f"Expected one of columns {candidates}, got {list(df.columns)}")


def annotate(results, entities):
    pdf = results.to_pandas()
    if not isinstance(pdf, pd.DataFrame):
        pdf = pd.DataFrame(pdf)

    pdf["vertex"] = pdf["vertex"].astype(int)
    pdf["name"] = pdf["vertex"].map(lambda vertex: entities[vertex]["name"])
    pdf["kind"] = pdf["vertex"].map(lambda vertex: entities[vertex]["kind"])

    if "distance" not in pdf:
        pdf["distance"] = -1
    pdf["distance"] = pdf["distance"].fillna(-1).astype(int)

    def distance_boost(distance):
        if distance < 0:
            return 0.0
        return 1.0 / (distance + 1)

    pdf["case_score"] = (pdf["pagerank"] * 100.0) + (
        pdf["distance"].map(distance_boost) * 10.0
    )
    return pdf


def print_table(title, pdf, columns, rows=12):
    print()
    print(title)
    print("-" * len(title))
    if len(pdf) == 0:
        print("(no rows)")
        return
    print(pdf.loc[:, columns].head(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seed-vertex",
        type=int,
        default=0,
        help="Known-bad entity to use as the BFS investigation seed.",
    )
    parser.add_argument(
        "--benign-components",
        type=int,
        default=6,
        help="Number of small disconnected benign components to add.",
    )
    args = parser.parse_args()

    entities, edge_df = build_synthetic_fraud_data(args.benign_components)
    if args.seed_vertex not in entities:
        sys.exit(f"FAIL: seed vertex {args.seed_vertex} is not in the synthetic graph")

    directed_graph = cugraph.Graph(directed=True)
    directed_graph.from_cudf_edgelist(
        edge_df, source="src", destination="dst", edge_attr="weight"
    )

    undirected_graph = cugraph.Graph(directed=False)
    undirected_graph.from_cudf_edgelist(
        edge_df, source="src", destination="dst", edge_attr="weight"
    )

    pagerank = cugraph.pagerank(directed_graph)
    bfs = cugraph.bfs(undirected_graph, start=args.seed_vertex)
    components = component_labels(undirected_graph)
    component_col = require_column(components, ["labels", "label", "component"])

    results = pagerank.merge(components, on="vertex", how="left").merge(
        bfs[[col for col in ["vertex", "distance", "predecessor"] if col in bfs.columns]],
        on="vertex",
        how="left",
    )
    pdf = annotate(results, entities)
    pdf = pdf.rename(columns={component_col: "component"})

    seed_name = entities[args.seed_vertex]["name"]
    print("Synthetic financial-services entity graph")
    print("=" * 43)
    print(f"Vertices:        {len(entities):,}")
    print(f"Edges:           {len(edge_df):,}")
    print(f"Known-bad seed:  {args.seed_vertex} ({seed_name})")

    top_risk = pdf.sort_values("case_score", ascending=False)
    print_table(
        "Investigation queue ranked by PageRank + BFS proximity",
        top_risk,
        ["vertex", "name", "kind", "pagerank", "distance", "component", "case_score"],
        rows=12,
    )

    reachable = pdf[pdf["distance"] >= 0].sort_values(
        ["distance", "case_score"], ascending=[True, False]
    )
    print_table(
        f"Entities reachable from {seed_name}",
        reachable,
        ["vertex", "name", "kind", "distance", "component"],
        rows=20,
    )

    component_sizes = (
        pdf.groupby("component")
        .agg(vertices=("vertex", "count"), avg_pagerank=("pagerank", "mean"))
        .sort_values(["vertices", "avg_pagerank"], ascending=False)
        .reset_index()
    )
    print_table(
        "Connected components",
        component_sizes,
        ["component", "vertices", "avg_pagerank"],
        rows=10,
    )

    suspicious_component = int(
        pdf.loc[pdf["vertex"] == args.seed_vertex, "component"].iloc[0]
    )
    suspicious_members = pdf[pdf["component"] == suspicious_component].sort_values(
        "case_score", ascending=False
    )
    print_table(
        "Suspicious component members",
        suspicious_members,
        ["vertex", "name", "kind", "pagerank", "distance", "case_score"],
        rows=20,
    )

    print()
    print(
        "Workflow fit: graph features such as component id, path distance, and "
        "PageRank can be joined back to transaction rows for analyst review, "
        "rules, or an ML fraud model."
    )


if __name__ == "__main__":
    main()
