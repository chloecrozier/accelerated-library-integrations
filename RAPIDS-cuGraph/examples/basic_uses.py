"""Basic cuGraph usage on a small GPU-resident graph.

This example starts with an edge list in cuDF, builds cuGraph graph objects,
then runs three common graph analytics operations:

1. PageRank: identify influential vertices in a directed graph.
2. BFS: find shortest unweighted paths from a starting vertex.
3. Weakly connected components: group vertices that belong to the same island.
"""

import sys


try:
    import cudf
    import cugraph
except ImportError as exc:
    sys.exit(
        "This example needs RAPIDS cuDF and cuGraph on a CUDA-capable Linux "
        f"or WSL2 machine -> {exc}"
    )


def print_section(title):
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def main():
    edges = cudf.DataFrame(
        {
            "src": [0, 1, 2, 2, 3, 4, 5, 6],
            "dst": [1, 2, 0, 3, 4, 2, 6, 5],
            "weight": [1.0, 1.0, 1.0, 0.7, 0.7, 0.4, 1.0, 1.0],
        }
    )

    print_section("Input edge list")
    print(edges)

    directed_graph = cugraph.Graph(directed=True)
    directed_graph.from_cudf_edgelist(
        edges, source="src", destination="dst", edge_attr="weight"
    )

    print_section("PageRank on directed graph")
    pagerank = cugraph.pagerank(directed_graph)
    print(pagerank.sort_values("pagerank", ascending=False))

    print_section("BFS from vertex 0")
    bfs = cugraph.bfs(directed_graph, start=0)
    print(bfs.sort_values("vertex"))

    undirected_graph = cugraph.Graph(directed=False)
    undirected_graph.from_cudf_edgelist(
        edges, source="src", destination="dst", edge_attr="weight"
    )

    print_section("Weakly connected components")
    components = cugraph.weakly_connected_components(undirected_graph)
    print(components.sort_values(["labels", "vertex"]))

    print()
    print("Done: cuDF edge data moved through cuGraph analytics successfully.")


if __name__ == "__main__":
    main()
