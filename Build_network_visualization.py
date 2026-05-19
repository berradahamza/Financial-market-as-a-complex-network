"""
Financial Network Builder and Visualizer

Input:
- yahoo_close_prices_10y_USA.csv

Output:
- financial_network_edges.csv
- financial_network_nodes.csv
- financial_network.graphml
- financial_network.gexf
- financial_network_preview.png

Method:
- Compute daily log-returns
- Compute correlation matrix
- Convert correlation to distance:
    d(i,j) = sqrt(2 * (1 - corr(i,j)))
- Build MST backbone
- Add strong correlation edges
- Detect communities with Louvain
- Export files for Gephi / report / visualization
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from scipy.sparse.csgraph import minimum_spanning_tree


# ============================================================
# 1. CONFIGURATION
# ============================================================

PRICE_FILE = "yahoo_close_prices_10y_USA.csv"

NODE_OUTPUT = "financial_network_nodes.csv"
EDGE_OUTPUT = "financial_network_edges.csv"

GRAPHML_OUTPUT = "financial_network.graphml"
GEXF_OUTPUT = "financial_network.gexf"

PREVIEW_OUTPUT = "financial_network_preview.png"

CORRELATION_THRESHOLD = 0.70

MIN_VALID_RETURN_RATIO = 0.90

MAX_NODES_FOR_PREVIEW = 5000
RANDOM_SEED = 42


# ============================================================
# 2. LOAD PRICE DATA
# ============================================================

def load_prices():
    prices = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)

    prices = prices.sort_index()

    print("Price matrix loaded:")
    print(prices.shape)

    return prices


# ============================================================
# 3. COMPUTE RETURNS
# ============================================================

def compute_log_returns(prices):
    log_prices = np.log(prices)
    returns = log_prices.diff().dropna(how="all")

    min_valid_values = int(len(returns) * MIN_VALID_RETURN_RATIO)
    returns = returns.dropna(axis=1, thresh=min_valid_values)

    returns = returns.fillna(returns.mean())

    print("\nReturns matrix:")
    print(returns.shape)

    return returns


# ============================================================
# 4. CORRELATION AND DISTANCE
# ============================================================

def compute_correlation_and_distance(returns):
    corr = returns.corr()

    corr = corr.replace([np.inf, -np.inf], np.nan)
    corr = corr.fillna(0)

    corr_values = corr.values
    corr_values = np.clip(corr_values, -1, 1)

    distance = np.sqrt(2 * (1 - corr_values))

    np.fill_diagonal(distance, 0)

    print("\nCorrelation and distance matrices computed.")
    print(f"Number of companies: {corr.shape[0]}")

    return corr, distance


# ============================================================
# 5. BUILD MST + STRONG CORRELATION NETWORK
# ============================================================

def build_hybrid_network(corr, distance):
    tickers = list(corr.columns)

    print("\nBuilding Minimum Spanning Tree...")

    mst_sparse = minimum_spanning_tree(distance)
    mst_coo = mst_sparse.tocoo()

    G = nx.Graph()

    for ticker in tickers:
        G.add_node(ticker)

    # Add MST edges
    for i, j, dist in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        ticker_i = tickers[i]
        ticker_j = tickers[j]

        correlation = corr.iloc[i, j]

        G.add_edge(
            ticker_i,
            ticker_j,
            weight=float(correlation),
            distance=float(dist),
            edge_type="MST"
        )

    print(f"MST edges added: {G.number_of_edges()}")

    print(f"\nAdding strong correlation edges: corr > {CORRELATION_THRESHOLD}")

    corr_values = corr.values
    n = len(tickers)

    added_strong_edges = 0

    for i in range(n):
        for j in range(i + 1, n):
            correlation = corr_values[i, j]

            if correlation > CORRELATION_THRESHOLD:
                ticker_i = tickers[i]
                ticker_j = tickers[j]

                if G.has_edge(ticker_i, ticker_j):
                    G[ticker_i][ticker_j]["edge_type"] = "MST+Strong"
                else:
                    dist = distance[i, j]

                    G.add_edge(
                        ticker_i,
                        ticker_j,
                        weight=float(correlation),
                        distance=float(dist),
                        edge_type="Strong"
                    )

                    added_strong_edges += 1

    print(f"Strong correlation edges added: {added_strong_edges}")
    print(f"Final network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


# ============================================================
# 6. COMMUNITY DETECTION AND CENTRALITIES
# ============================================================

def add_network_measures(G):
    print("\nComputing centralities and communities...")

    degree_dict = dict(G.degree())
    weighted_degree_dict = dict(G.degree(weight="weight"))

    betweenness_dict = nx.betweenness_centrality(G, k=min(300, G.number_of_nodes()), seed=RANDOM_SEED)
    eigenvector_dict = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")

    communities = nx.community.louvain_communities(G, weight="weight", seed=RANDOM_SEED)

    community_dict = {}
    for community_id, community_nodes in enumerate(communities):
        for node in community_nodes:
            community_dict[node] = community_id

    nx.set_node_attributes(G, degree_dict, "degree")
    nx.set_node_attributes(G, weighted_degree_dict, "weighted_degree")
    nx.set_node_attributes(G, betweenness_dict, "betweenness")
    nx.set_node_attributes(G, eigenvector_dict, "eigenvector")
    nx.set_node_attributes(G, community_dict, "community")

    print(f"Detected communities: {len(communities)}")

    return G


# ============================================================
# 7. EXPORT NODE AND EDGE TABLES
# ============================================================

def export_tables(G):
    nodes = []

    for node, data in G.nodes(data=True):
        nodes.append({
            "ticker": node,
            "degree": data.get("degree"),
            "weighted_degree": data.get("weighted_degree"),
            "betweenness": data.get("betweenness"),
            "eigenvector": data.get("eigenvector"),
            "community": data.get("community")
        })

    nodes_df = pd.DataFrame(nodes)
    nodes_df = nodes_df.sort_values("degree", ascending=False)
    nodes_df.to_csv(NODE_OUTPUT, index=False)

    edges = []

    for source, target, data in G.edges(data=True):
        edges.append({
            "source": source,
            "target": target,
            "correlation": data.get("weight"),
            "distance": data.get("distance"),
            "edge_type": data.get("edge_type")
        })

    edges_df = pd.DataFrame(edges)
    edges_df = edges_df.sort_values("correlation", ascending=False)
    edges_df.to_csv(EDGE_OUTPUT, index=False)

    print("\nTables exported:")
    print(f"- {NODE_OUTPUT}")
    print(f"- {EDGE_OUTPUT}")


# ============================================================
# 8. EXPORT GRAPH FILES
# ============================================================

def export_graph_files(G):
    nx.write_graphml(G, GRAPHML_OUTPUT)
    nx.write_gexf(G, GEXF_OUTPUT)

    print("\nGraph files exported:")
    print(f"- {GRAPHML_OUTPUT}")
    print(f"- {GEXF_OUTPUT}")


# ============================================================
# 9. QUICK PREVIEW VISUALIZATION
# ============================================================

def create_preview(G):
    print("\nCreating preview visualization...")

    if G.number_of_nodes() > MAX_NODES_FOR_PREVIEW:
        top_nodes = sorted(
            G.nodes(),
            key=lambda n: G.nodes[n]["degree"],
            reverse=True
        )[:MAX_NODES_FOR_PREVIEW]

        H = G.subgraph(top_nodes).copy()
    else:
        H = G.copy()

    pos = nx.spring_layout(H, seed=RANDOM_SEED, weight="weight", iterations=80)

    communities = [H.nodes[n].get("community", 0) for n in H.nodes()]
    node_sizes = [20 + H.nodes[n].get("degree", 1) * 3 for n in H.nodes()]

    plt.figure(figsize=(16, 12))

    nx.draw_networkx_edges(
        H,
        pos,
        alpha=0.15,
        width=0.5
    )

    nx.draw_networkx_nodes(
        H,
        pos,
        node_size=node_sizes,
        node_color=communities,
        cmap=plt.cm.tab20,
        alpha=0.85
    )

    plt.title("Financial Correlation Network Preview\nMST Backbone + Strong Correlation Edges")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PREVIEW_OUTPUT, dpi=300)
    plt.close()

    print(f"Preview saved: {PREVIEW_OUTPUT}")


# ============================================================
# 10. NETWORK SUMMARY
# ============================================================

def print_network_summary(G):
    print("\nNetwork summary:")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.6f}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.4f}")
    print(f"Average clustering: {nx.average_clustering(G):.4f}")

    if nx.is_connected(G):
        print("Connected: Yes")
        print(f"Average shortest path length: {nx.average_shortest_path_length(G):.4f}")
        print(f"Diameter: {nx.diameter(G)}")
    else:
        print("Connected: No")
        largest_cc = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest_cc)
        print(f"Largest component size: {H.number_of_nodes()}")


# ============================================================
# 11. RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    prices = load_prices()

    returns = compute_log_returns(prices)

    corr, distance = compute_correlation_and_distance(returns)

    G = build_hybrid_network(corr, distance)

    G = add_network_measures(G)

    print_network_summary(G)

    export_tables(G)

    export_graph_files(G)

    create_preview(G)

    print("\nDone.")