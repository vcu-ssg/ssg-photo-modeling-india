
import networkx as nx

# scripts/convert_matches_g_to_dot.py

def convert_matches_g_to_dot(input_path: str, output_path: str) -> None:
    """
    Convert OpenMVG matches.g.txt (edge list) to DOT format.

    Args:
        input_path (str): Path to matches.g.txt
        output_path (str): Path to output DOT file
    """
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        fout.write("graph G {\n")
        for line in fin:
            tokens = line.strip().split()
            if len(tokens) != 2:
                continue
            src, dst = tokens
            fout.write(f"    {src} -- {dst};\n")
        fout.write("}\n")


def analyze_graph(matches_file, show_disconnected):
    """
    Analyze matches graph (.g.txt) and print quality report.
    """
    print(f"📂 Loading matches file: {matches_file}")

    # Read pairs
    edges = []
    with open(matches_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            a, b = map(int, parts)
            edges.append((a, b))

    # Build graph
    G = nx.Graph()
    G.add_edges_from(edges)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    components = list(nx.connected_components(G))
    largest_component_size = max(len(c) for c in components) if components else 0

    disconnected_nodes = [node for node in G.nodes if G.degree[node] == 0]

    # Report
    print("\n📊 Graph Quality Report")
    print("-----------------------")
    print(f"Total nodes          : {num_nodes}")
    print(f"Total edges          : {num_edges}")
    print(f"Connected components : {len(components)}")
    print(f"Largest component    : {largest_component_size} nodes")
    print(f"Disconnected nodes   : {len(disconnected_nodes)}")

    if show_disconnected and disconnected_nodes:
        print("\n🚫 Disconnected nodes:")
        print(disconnected_nodes)

    print("\n✅ Done.")
    