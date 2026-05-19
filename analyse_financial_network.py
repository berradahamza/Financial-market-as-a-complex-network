"""
Financial Network Structural Analysis
Project: Financial Market as a Complex Network

Input:
- financial_network.graphml
or
- financial_network.gexf

Expected graph attributes from the builder:
Nodes:
- degree
- weighted_degree
- betweenness
- eigenvector
- community

Edges:
- weight = correlation
- distance = correlation distance
- edge_type = MST / Strong / MST+Strong

Output:
- analysis_outputs/network_summary.csv
- analysis_outputs/top_central_nodes.csv
- analysis_outputs/community_summary.csv
- analysis_outputs/null_model_comparison_detailed.csv
- analysis_outputs/null_model_comparison_summary.csv
- analysis_outputs/robustness_results.csv
- analysis_outputs/degree_distribution.png
- analysis_outputs/correlation_distribution.png
- analysis_outputs/community_size_distribution.png
- analysis_outputs/robustness_curves.png
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

GRAPH_FILE = "financial_network.graphml"
OUTPUT_DIR = "analysis_outputs"

RANDOM_SEED = 42
APPROX_BETWEENNESS_K = 300
ROBUSTNESS_STEPS = 20

# Increased from 3 to 30 for more stable null-model results
NULL_MODEL_REPETITIONS = 30

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")


# ============================================================
# 2. LOAD GRAPH
# ============================================================

def load_graph(path):
    if path.endswith(".graphml"):
        G = nx.read_graphml(path)
    elif path.endswith(".gexf"):
        G = nx.read_gexf(path)
    else:
        raise ValueError("Unsupported graph format. Use .graphml or .gexf")

    G = nx.Graph(G)

    # Convert numeric node attributes loaded as strings
    for n, data in G.nodes(data=True):
        for attr in ["degree", "weighted_degree", "betweenness", "eigenvector", "community"]:
            if attr in data:
                try:
                    if attr == "community":
                        data[attr] = int(float(data[attr]))
                    else:
                        data[attr] = float(data[attr])
                except Exception:
                    data[attr] = np.nan

    # Convert numeric edge attributes loaded as strings
    for u, v, data in G.edges(data=True):
        for attr in ["weight", "distance"]:
            if attr in data:
                try:
                    data[attr] = float(data[attr])
                except Exception:
                    data[attr] = np.nan

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ============================================================
# 3. BASIC STRUCTURAL DESCRIPTORS
# ============================================================

def safe_average_shortest_path_and_diameter(G):
    """
    Computes average shortest path length and diameter on the largest connected component.
    For large graphs, diameter is approximated to avoid excessive runtime.
    """

    if G.number_of_nodes() == 0:
        return np.nan, np.nan, 0

    if nx.is_connected(G):
        H = G
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest_cc).copy()

    if H.number_of_nodes() <= 1:
        return 0, 0, H.number_of_nodes()

    avg_path = nx.average_shortest_path_length(H)

    if H.number_of_nodes() <= 1500:
        diameter = nx.diameter(H)
    else:
        diameter = nx.approximation.diameter(H, seed=RANDOM_SEED)

    return avg_path, diameter, H.number_of_nodes()


def compute_basic_summary(G):
    degrees = dict(G.degree())
    weighted_degrees = dict(G.degree(weight="weight"))

    avg_path, diameter, largest_cc_size = safe_average_shortest_path_and_diameter(G)

    summary = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "min_degree": min(degrees.values()),
        "max_degree": max(degrees.values()),
        "average_degree": np.mean(list(degrees.values())),
        "average_weighted_degree": np.mean(list(weighted_degrees.values())),
        "average_clustering_unweighted": nx.average_clustering(G),
        "average_clustering_weighted": nx.average_clustering(G, weight="weight"),
        "assortativity_degree": nx.degree_assortativity_coefficient(G),
        "connected": nx.is_connected(G),
        "largest_connected_component_size": largest_cc_size,
        "average_shortest_path_lcc": avg_path,
        "diameter_lcc": diameter,
    }

    df = pd.DataFrame([summary])
    df.to_csv(os.path.join(OUTPUT_DIR, "network_summary.csv"), index=False)

    print("\nBasic network summary:")
    print(df.T)

    return summary


# ============================================================
# 4. CENTRALITY ANALYSIS
# ============================================================

def compute_centralities(G):
    """
    Computes centrality metrics.
    Betweenness is approximated for large graphs.
    """

    print("\nComputing centralities...")

    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))

    betweenness = nx.betweenness_centrality(
        G,
        k=min(APPROX_BETWEENNESS_K, G.number_of_nodes()),
        seed=RANDOM_SEED,
        weight=None
    )

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except Exception:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")

    rows = []
    for node in G.nodes():
        rows.append({
            "ticker": node,
            "degree": degree.get(node, 0),
            "weighted_degree": weighted_degree.get(node, 0),
            "betweenness": betweenness.get(node, 0),
            "eigenvector": eigenvector.get(node, 0),
            "community": G.nodes[node].get("community", np.nan),
        })

    df = pd.DataFrame(rows)

    df["rank_degree"] = df["degree"].rank(ascending=False, method="min")
    df["rank_weighted_degree"] = df["weighted_degree"].rank(ascending=False, method="min")
    df["rank_betweenness"] = df["betweenness"].rank(ascending=False, method="min")
    df["rank_eigenvector"] = df["eigenvector"].rank(ascending=False, method="min")

    df = df.sort_values("degree", ascending=False)
    df.to_csv(os.path.join(OUTPUT_DIR, "top_central_nodes.csv"), index=False)

    print("\nTop 10 nodes by degree:")
    print(df[["ticker", "degree", "weighted_degree", "betweenness", "eigenvector", "community"]].head(10))

    return df


# ============================================================
# 5. COMMUNITY ANALYSIS
# ============================================================

def compute_communities(G):
    """
    Uses existing Louvain community labels if available.
    Otherwise computes Louvain communities.
    """

    print("\nAnalyzing communities...")

    has_communities = all("community" in G.nodes[n] for n in G.nodes())

    if has_communities:
        community_dict = {}
        for node in G.nodes():
            comm = int(G.nodes[node]["community"])
            community_dict.setdefault(comm, set()).add(node)
        communities = list(community_dict.values())
    else:
        communities = nx.community.louvain_communities(G, weight="weight", seed=RANDOM_SEED)
        for cid, nodes in enumerate(communities):
            for n in nodes:
                G.nodes[n]["community"] = cid

    modularity_weighted = nx.community.modularity(G, communities, weight="weight")
    modularity_unweighted = nx.community.modularity(G, communities, weight=None)

    rows = []
    for cid, nodes in enumerate(communities):
        sub = G.subgraph(nodes).copy()

        internal_edges = sub.number_of_edges()
        possible_internal_edges = len(nodes) * (len(nodes) - 1) / 2
        internal_density = internal_edges / possible_internal_edges if possible_internal_edges > 0 else 0

        avg_internal_weight = np.mean([
            data.get("weight", 0)
            for _, _, data in sub.edges(data=True)
        ]) if sub.number_of_edges() > 0 else 0

        top_nodes = sorted(
            nodes,
            key=lambda n: G.degree(n),
            reverse=True
        )[:5]

        rows.append({
            "community": cid,
            "size": len(nodes),
            "internal_edges": internal_edges,
            "internal_density": internal_density,
            "average_internal_correlation": avg_internal_weight,
            "top_degree_tickers": ", ".join(top_nodes),
        })

    df = pd.DataFrame(rows).sort_values("size", ascending=False)

    df["weighted_modularity_global"] = modularity_weighted
    df["unweighted_modularity_global"] = modularity_unweighted

    df.to_csv(os.path.join(OUTPUT_DIR, "community_summary.csv"), index=False)

    print(f"\nDetected communities: {len(communities)}")
    print(f"Weighted modularity: {modularity_weighted:.4f}")
    print(f"Unweighted modularity: {modularity_unweighted:.4f}")
    print("\nLargest communities:")
    print(df.head(10))

    return communities, df


# ============================================================
# 6. IMPROVED NULL MODEL COMPARISON
# ============================================================

def has_valid_weights(G):
    """
    Returns True only if at least one edge has a valid numeric weight.
    """
    weights = []
    for _, _, data in G.edges(data=True):
        w = data.get("weight", None)
        if w is not None:
            try:
                weights.append(float(w))
            except Exception:
                pass

    return len(weights) > 0


def compute_louvain_metrics(G, use_weight=False):
    """
    Computes Louvain communities and modularity.
    For fair null-model comparison, unweighted modularity is preferred.
    Weighted modularity should only be used when edge weights have a real interpretation.
    """

    weight_arg = "weight" if use_weight else None

    communities = nx.community.louvain_communities(
        G,
        weight=weight_arg,
        seed=RANDOM_SEED
    )

    modularity = nx.community.modularity(
        G,
        communities,
        weight=weight_arg
    )

    return len(communities), modularity


def summarize_graph_for_comparison(
    G,
    network_name,
    model_type,
    repetition,
    original_degree_sequence=None
):
    """
    Computes structural metrics for the observed graph or a null model.
    """

    avg_path, diameter, largest_cc_size = safe_average_shortest_path_and_diameter(G)

    degrees = [d for _, d in G.degree()]

    # Fair comparison across all graphs
    k_communities_unweighted, modularity_unweighted = compute_louvain_metrics(
        G,
        use_weight=False
    )

    # Weighted modularity only when edge weights exist and are meaningful
    if has_valid_weights(G):
        k_communities_weighted, modularity_weighted = compute_louvain_metrics(
            G,
            use_weight=True
        )
    else:
        k_communities_weighted = np.nan
        modularity_weighted = np.nan

    # Configuration model diagnostic:
    # after simplifying a multigraph, the degree sequence is no longer perfectly preserved.
    degree_sequence_error_mean = np.nan
    degree_sequence_error_max = np.nan

    if original_degree_sequence is not None:
        generated_degrees = sorted([d for _, d in G.degree()])
        original_degrees = sorted(original_degree_sequence)

        min_len = min(len(generated_degrees), len(original_degrees))

        errors = [
            abs(generated_degrees[i] - original_degrees[i])
            for i in range(min_len)
        ]

        degree_sequence_error_mean = np.mean(errors)
        degree_sequence_error_max = np.max(errors)

    return {
        "network": network_name,
        "model_type": model_type,
        "repetition": repetition,

        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),

        "min_degree": min(degrees),
        "max_degree": max(degrees),
        "average_degree": np.mean(degrees),

        "average_clustering": nx.average_clustering(G),
        "assortativity": nx.degree_assortativity_coefficient(G),

        "connected": nx.is_connected(G),
        "largest_cc_size": largest_cc_size,
        "average_path_lcc": avg_path,
        "diameter_lcc": diameter,

        "louvain_communities_unweighted": k_communities_unweighted,
        "modularity_unweighted": modularity_unweighted,

        "louvain_communities_weighted": k_communities_weighted,
        "modularity_weighted": modularity_weighted,

        "degree_sequence_error_mean": degree_sequence_error_mean,
        "degree_sequence_error_max": degree_sequence_error_max,
    }


def generate_configuration_simple_graph(degree_sequence, seed):
    """
    Generates a simplified Configuration Model graph.

    Important:
    NetworkX configuration_model creates a multigraph with self-loops.
    After converting to a simple Graph and removing self-loops, the degree sequence is
    only approximately preserved.
    """

    G_multi = nx.configuration_model(degree_sequence, seed=seed)

    G_simple = nx.Graph(G_multi)
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))

    # Relabel nodes as strings for consistency with loaded GraphML tickers not needed here,
    # but keeps output clean.
    G_simple = nx.convert_node_labels_to_integers(G_simple)

    return G_simple


def compare_with_null_models(G):
    """
    Compares the observed financial network with three null models:

    1. ER same n,m:
       Controls only for number of nodes and edges.

    2. BA similar average degree:
       Tests whether a hub-driven preferential attachment mechanism can reproduce the structure.

    3. Configuration Model:
       Controls approximately for the empirical degree sequence.

    The main fair comparison uses unweighted metrics for all graphs.
    Weighted modularity is reported separately for the financial network because only its edge
    weights have a financial interpretation.
    """

    print("\nComparing with null models...")

    n = G.number_of_nodes()
    m = G.number_of_edges()
    degree_sequence = [d for _, d in G.degree()]
    avg_degree = np.mean(degree_sequence)

    results = []

    # Observed financial network
    results.append(
        summarize_graph_for_comparison(
            G=G,
            network_name="Financial network",
            model_type="Financial network",
            repetition=0
        )
    )

    for rep in range(1, NULL_MODEL_REPETITIONS + 1):
        seed = RANDOM_SEED + rep

        print(f"Null model repetition {rep}/{NULL_MODEL_REPETITIONS}")

        # ----------------------------------------------------
        # ER null model: same number of nodes and edges
        # ----------------------------------------------------
        G_er = nx.gnm_random_graph(n, m, seed=seed)

        results.append(
            summarize_graph_for_comparison(
                G=G_er,
                network_name=f"ER same n,m rep {rep}",
                model_type="ER same n,m",
                repetition=rep
            )
        )

        # ----------------------------------------------------
        # BA null model: similar average degree
        # Average degree in BA is approximately 2m_ba.
        # ----------------------------------------------------
        m_ba = max(1, int(round(avg_degree / 2)))

        G_ba = nx.barabasi_albert_graph(n, m_ba, seed=seed)

        results.append(
            summarize_graph_for_comparison(
                G=G_ba,
                network_name=f"BA similar avg degree rep {rep}",
                model_type="BA similar avg degree",
                repetition=rep
            )
        )

        # ----------------------------------------------------
        # Configuration Model: approximately same degree sequence
        # ----------------------------------------------------
        G_conf = generate_configuration_simple_graph(degree_sequence, seed=seed)

        results.append(
            summarize_graph_for_comparison(
                G=G_conf,
                network_name=f"Configuration model rep {rep}",
                model_type="Configuration model",
                repetition=rep,
                original_degree_sequence=degree_sequence
            )
        )

    detailed_df = pd.DataFrame(results)

    detailed_path = os.path.join(OUTPUT_DIR, "null_model_comparison_detailed.csv")
    detailed_df.to_csv(detailed_path, index=False)

    print(f"\nDetailed null-model comparison saved to: {detailed_path}")
    print(detailed_df.head())

    # Build summary table: mean and std for null models
    metrics_to_summarize = [
        "nodes",
        "edges",
        "density",
        "min_degree",
        "max_degree",
        "average_degree",
        "average_clustering",
        "assortativity",
        "largest_cc_size",
        "average_path_lcc",
        "diameter_lcc",
        "louvain_communities_unweighted",
        "modularity_unweighted",
        "degree_sequence_error_mean",
        "degree_sequence_error_max",
    ]

    null_only = detailed_df[detailed_df["model_type"] != "Financial network"].copy()

    summary_mean = null_only.groupby("model_type")[metrics_to_summarize].mean()
    summary_std = null_only.groupby("model_type")[metrics_to_summarize].std()

    summary_df = pd.DataFrame()

    for metric in metrics_to_summarize:
        summary_df[f"{metric}_mean"] = summary_mean[metric]
        summary_df[f"{metric}_std"] = summary_std[metric]

    summary_df = summary_df.reset_index()

    # Add observed financial network values for easy comparison
    financial_row = detailed_df[detailed_df["model_type"] == "Financial network"].iloc[0]

    for metric in metrics_to_summarize:
        summary_df[f"{metric}_financial"] = financial_row[metric]

    summary_path = os.path.join(OUTPUT_DIR, "null_model_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Aggregated null-model summary saved to: {summary_path}")
    print(summary_df)

    return detailed_df, summary_df


# ============================================================
# 7. ROBUSTNESS ANALYSIS
# ============================================================

def largest_component_fraction(G):
    if G.number_of_nodes() == 0:
        return 0

    largest_cc = max(nx.connected_components(G), key=len)
    return len(largest_cc) / G.number_of_nodes()


def robustness_curve(G, removal_order, label):
    H = G.copy()
    n0 = H.number_of_nodes()

    fractions_removed = []
    lcc_fractions = []

    steps = np.linspace(0, len(removal_order), ROBUSTNESS_STEPS + 1, dtype=int)

    removed_until = 0

    for step in steps:
        to_remove = removal_order[removed_until:step]
        H.remove_nodes_from(to_remove)
        removed_until = step

        fractions_removed.append(step / n0)
        lcc_fractions.append(largest_component_fraction(H))

    return pd.DataFrame({
        "strategy": label,
        "fraction_removed": fractions_removed,
        "largest_component_fraction": lcc_fractions
    })


def run_robustness_analysis(G, centrality_df):
    """
    Compares random failure with targeted attacks on central nodes.
    """

    print("\nRunning robustness analysis...")

    degree_order = centrality_df.sort_values("degree", ascending=False)["ticker"].tolist()
    betweenness_order = centrality_df.sort_values("betweenness", ascending=False)["ticker"].tolist()

    random_order = list(G.nodes())
    random.shuffle(random_order)

    df_degree = robustness_curve(G, degree_order, "targeted_degree")
    df_betweenness = robustness_curve(G, betweenness_order, "targeted_betweenness")
    df_random = robustness_curve(G, random_order, "random_failure")

    df = pd.concat([df_degree, df_betweenness, df_random], ignore_index=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "robustness_results.csv"), index=False)

    plt.figure(figsize=(10, 6))

    for strategy, group in df.groupby("strategy"):
        plt.plot(
            group["fraction_removed"],
            group["largest_component_fraction"],
            marker="o",
            label=strategy
        )

    plt.title("Robustness of the Financial Correlation Network")
    plt.xlabel("Fraction of removed nodes")
    plt.ylabel("Largest connected component fraction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "robustness_curves.png"), dpi=300)
    plt.close()

    print("Robustness results saved.")

    return df


# ============================================================
# 8. PLOTS
# ============================================================

def plot_degree_distribution(G):
    degrees = np.array([d for _, d in G.degree()])

    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=50)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "degree_distribution.png"), dpi=300)
    plt.close()

    degree_counts = pd.Series(degrees).value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.loglog(degree_counts.index, degree_counts.values, marker="o", linestyle="none")
    plt.title("Degree Distribution in Log-Log Scale")
    plt.xlabel("Degree k")
    plt.ylabel("Frequency P(k)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "degree_distribution_loglog.png"), dpi=300)
    plt.close()


def plot_correlation_distribution(G):
    weights = np.array([
        data.get("weight", np.nan)
        for _, _, data in G.edges(data=True)
    ])

    weights = weights[~np.isnan(weights)]

    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=50)
    plt.title("Edge Correlation Distribution")
    plt.xlabel("Correlation")
    plt.ylabel("Number of edges")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_distribution.png"), dpi=300)
    plt.close()


def plot_community_sizes(community_df):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(community_df)), community_df["size"])
    plt.title("Community Size Distribution")
    plt.xlabel("Community rank by size")
    plt.ylabel("Number of nodes")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "community_size_distribution.png"), dpi=300)
    plt.close()


# ============================================================
# 9. IMPROVED INTERPRETATION HELPER
# ============================================================

def write_interpretation_template(summary, null_detailed_df, null_summary_df, community_df, centrality_df):
    """
    Writes report-ready interpretation notes.
    """

    financial = null_detailed_df[null_detailed_df["model_type"] == "Financial network"].iloc[0]

    top_degree_nodes = ", ".join(
        centrality_df.sort_values("degree", ascending=False)["ticker"].head(10).tolist()
    )

    top_betweenness_nodes = ", ".join(
        centrality_df.sort_values("betweenness", ascending=False)["ticker"].head(10).tolist()
    )

    text = f"""
Financial Network Interpretation Notes

1. Scale of the network
The final graph contains {int(summary["nodes"])} nodes and {int(summary["edges"])} edges.
This is large enough for a complex network study and is comparable in scale to large empirical networks
studied in class.

2. Density and sparsity
The density is {summary["density"]:.6f}, which shows that the graph is sparse.
This is expected because the construction keeps the MST backbone and only adds strong correlation edges,
instead of keeping all possible pairwise correlations.

3. Clustering and communities
The average unweighted clustering coefficient is {summary["average_clustering_unweighted"]:.4f}.
The Louvain algorithm detects {len(community_df)} communities.
The weighted modularity is {community_df["weighted_modularity_global"].iloc[0]:.4f}.
The unweighted modularity is {community_df["unweighted_modularity_global"].iloc[0]:.4f}.

This suggests that the financial network is not homogeneous. It contains groups of securities that are
more strongly connected internally than externally.

4. Central securities
The top central nodes by degree are:
{top_degree_nodes}

These nodes have the largest number of direct connections in the filtered correlation network.
They can be interpreted as securities with broad co-movement links to many other securities.

5. Bridge securities
The top nodes by betweenness are:
{top_betweenness_nodes}

These nodes may act as bridges between different market communities.

6. Null-model comparison
The financial network is compared to three null models:

- ER graph with the same number of nodes and edges.
- BA graph with similar average degree.
- Configuration Model graph approximately preserving the empirical degree sequence.

The main fair comparison uses unweighted metrics, because the null models do not have meaningful financial
correlation weights. Weighted modularity is reported separately for the financial network.

Observed financial values:
- Average clustering: {financial["average_clustering"]:.4f}
- Degree assortativity: {financial["assortativity"]:.4f}
- Average path length on LCC: {financial["average_path_lcc"]:.4f}
- Unweighted modularity: {financial["modularity_unweighted"]:.4f}
- Number of Louvain communities: {int(financial["louvain_communities_unweighted"])}

Use null_model_comparison_summary.csv to compare these values against the mean and standard deviation
of ER, BA and Configuration Model graphs.

If the financial network has higher clustering and modularity than ER, then its structure is not explained
by random wiring alone.

If it differs strongly from BA, then its structure is not explained only by hub formation.

If it also differs from the Configuration Model, then its community structure is not only a consequence of
the observed degree sequence.

7. Important methodological limitation
The real financial network is built from a correlation-distance matrix using an MST backbone plus strong
correlation edges. The null models are generated directly as graphs. Therefore, the comparison tests whether
the final graph resembles standard random graph mechanisms, but it does not reproduce the full financial
network construction pipeline.

8. Robustness
Use robustness_curves.png to compare random failure and targeted attacks.
If targeted removal of high-degree or high-betweenness nodes fragments the graph faster than random removal,
the network depends strongly on structurally central securities.
"""

    path = os.path.join(OUTPUT_DIR, "interpretation_notes.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Interpretation notes saved to {path}")


# ============================================================
# 10. RUN FULL ANALYSIS
# ============================================================

if __name__ == "__main__":

    G = load_graph(GRAPH_FILE)

    summary = compute_basic_summary(G)

    centrality_df = compute_centralities(G)

    communities, community_df = compute_communities(G)

    null_detailed_df, null_summary_df = compare_with_null_models(G)

    robustness_df = run_robustness_analysis(G, centrality_df)

    plot_degree_distribution(G)
    plot_correlation_distribution(G)
    plot_community_sizes(community_df)

    write_interpretation_template(
        summary=summary,
        null_detailed_df=null_detailed_df,
        null_summary_df=null_summary_df,
        community_df=community_df,
        centrality_df=centrality_df
    )

    print("\nAnalysis completed.")
    print(f"All outputs saved in: {OUTPUT_DIR}")