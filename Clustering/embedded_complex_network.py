import numpy as np
from fastdtw import fastdtw
import pandas as pd
import community as community_louvain
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import (
    euclidean_distances,
    manhattan_distances,
    cosine_distances,
)
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import os
from datetime import datetime


## Evaluation Functions
def evaluate_partition(G: nx.Graph, partition: dict) -> float:
    """
    Evaluate the partition using modularity - Only for Louvain
    Input:
    G: nx.Graph - The graph object
    partition: dict - The partition dictionary
    Output:
    modularity: float - The modularity score
    """
    modularity = community_louvain.modularity(partition, G)
    return modularity


def calculate_modularity(G: nx.Graph, clusters: list) -> float:
    """
    Calculate modularity based on the clusters for non-Louvain algorithms
    Input:
    G: nx.Graph - The graph object
    clusters: list - The list of clusters
    Output:
    modularity: float - The modularity score

    """
    # Create a list of sets for communities
    communities = [set() for _ in range(max(clusters) + 1)]
    for node, cluster in enumerate(clusters):
        communities[cluster].add(node)
    # Check if all nodes are covered
    all_nodes = set(G.nodes())
    union_of_clusters = set.union(*communities)
    if all_nodes != union_of_clusters:
        raise ValueError("The union of all communities does not include all nodes.")

    # Calculate modularity
    modularity = nx.algorithms.community.modularity(G, communities)
    return modularity


def evaluate_partition_sil(data: np.ndarray, partition: dict) -> float:
    """
    Evaluate the partition using Silhouette Score
    Input:
    data: np.ndarray - The data used for clustering
    partition: dict - The partition dictionary
    Output:
    silhouette: float - The Silhouette Score
    """

    # Converting partition dictionary to a list of labels aligned with data
    labels = [partition.get(i, -1) for i in range(len(data))]

    # Check if there is more than one label and more than zero labels to calculate Silhouette Score
    if len(set(labels)) > 1 and len(set(labels)) < len(data):
        silhouette = silhouette_score(data, labels, metric="euclidean")
        return silhouette
    else:
        print(
            "Not enough clusters for Silhouette Score. There must be at least 2 clusters."
        )
        return None


def evaluate_davies_dbi(G: nx.Graph, partition: dict) -> float:
    """
    Evaluate the partition using Davies Bouldin Index
    Input:
    G: nx.Graph - The graph object
    partition: dict - The partition dictionary
    Output:
    dbi: float - The Davies Bouldin Index
    """

    # Create reverse lookup for clusters
    clusters = {}
    for node, cluster in partition.items():
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(node)

    # Step 1: Identify centroids (as medoids)
    centroids = {}
    for cluster, nodes in clusters.items():
        min_distance_sum = float("inf")
        centroid = nodes[0]
        for node in nodes:
            distances = nx.single_source_shortest_path_length(G, node)
            distance_sum = sum(distances[n] for n in nodes if n in distances)
            if distance_sum < min_distance_sum:
                min_distance_sum = distance_sum
                centroid = node
        centroids[cluster] = centroid

    # Step 2: Calculate intra-cluster distances and inter-cluster min distances
    intra_distances = {}
    for cluster, nodes in clusters.items():
        centroid = centroids[cluster]
        distances = nx.single_source_shortest_path_length(G, centroid)
        intra_distances[cluster] = np.mean(
            [distances[node] for node in nodes if node in distances]
        )

    # Step 3: Calculate DBI
    dbi_sum = 0
    for i in clusters:
        max_ratio = 0
        for j in clusters:
            if i != j:
                inter_distance = nx.shortest_path_length(
                    G, source=centroids[i], target=centroids[j]
                )
                ratio = (intra_distances[i] + intra_distances[j]) / inter_distance
                if ratio > max_ratio:
                    max_ratio = ratio
        dbi_sum += max_ratio

    return dbi_sum / len(clusters)


## Distance Functions
def dtw_distances(data: np.ndarray) -> np.ndarray:
    """
    Calculate DTW distances between data samples
    Input:
    data: np.ndarray - The data to calculate distances for
    Output:
    distances: np.ndarray - The pairwise DTW distances - Shape: (n, n) where n is the number of samples
    """
    n = len(data)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Ensure each data sample is treated as a 1-D vector
            distance, _ = fastdtw(
                data[i].reshape(-1, 1), data[j].reshape(-1, 1), dist=euclidean
            )
            distances[i, j] = distance
            distances[j, i] = distance
    return distances


### Clustering Algorithm
def girvan_newman(G: nx.Graph, num_communities=None, target_modularity=None) -> tuple:
    """
    Girvan-Newman algorithm for community detection
    Input:
    G: nx.Graph - The graph object
    num_communities: int - The number of communities to find - Provide either this or target_modularity
    target_modularity: float - The target modularity to reach
    Output:
    best_communities: list - The list of communities
    best_modularity: float - The modularity score
    """
    # Start with the initial graph
    original_graph = G.copy()
    # Track the best modularity
    best_modularity = -1
    best_communities = []

    while len(G.edges()) > 0:
        edge_betweenness = nx.edge_betweenness_centrality(G)
        max_edge = max(edge_betweenness, key=edge_betweenness.get)
        G.remove_edge(*max_edge)

        components = list(nx.connected_components(G))
        current_modularity = nx.algorithms.community.modularity(
            original_graph, components
        )

        # Check if current modularity is the best we've seen
        if current_modularity > best_modularity:
            best_modularity = current_modularity
            best_communities = components

        # Stopping condition on modularity
        if target_modularity and current_modularity > target_modularity:
            print(
                f"Stopping: Modularity {current_modularity} exceeds target {target_modularity}"
            )
            break

        # Stopping condition on number of communities
        if num_communities and len(components) >= num_communities:
            print(
                f"Stopping: Number of communities {len(components)} reached target {num_communities}"
            )
            break

    return best_communities, best_modularity


## Main Function
def main(
    distance_metric: str,
    connectivity: str,
    k_neighbors: int,
    graph_construction_method: str,
    percentile: float,
    clustering_algorithm: str,
    girvan_num_communities: int,
    girvan_target_modularity: float,
    spectral_num_clusters: int,
    model_name: str,
):
    """
    Main function to run community detection on generated data
    Input:
    distance_metric: str - The distance metric to use (euclidean, manhattan, cosine, dtw)
    connectivity: str - The method to create graph edges (epsilon for epsilon-neighborhood, knn for k-nearest neighbors)
    k_neighbors: int - Number of neighbors to use for k-NN method
    graph_construction_method: str - The method to construct the graph (flatten, aggregate)
    percentile: float - The percentile to use for epsilon-neighborhood method
    clustering_algorithm: str - The clustering algorithm to use (louvain, spectral, girvan_newman)
    girvan_num_communities: int - Number of communities to find with Girvan-Newman algorithm
    girvan_target_modularity: float - Target modularity to reach with Girvan-Newman algorithm
    spectral_num_clusters: int - Number of clusters to find with Spectral Clustering algorithm
    model_name: str - Name of the model used to generate the data
    """

    # Load encoded data
    data = np.load(f"../Generated/{model_name}/encoded_data.npy")

    print(f"Data shape: {data.shape}")
    print(f"Latent space shape: {data.shape[1]}")

    print("Loading Experiment for model: ", model_name)

    # Calculate pairwise distances based on the selected metric
    if distance_metric == "euclidean":
        print("Calculating Euclidean distances...")
        distances = euclidean_distances(data)
    elif distance_metric == "manhattan":
        print("Calculating Manhattan distances...")
        distances = manhattan_distances(data)
    elif distance_metric == "cosine":
        print("Calculating Cosine distances...")
        distances = cosine_distances(data)
    elif distance_metric == "dtw":
        print("Calculating DTW distances...")
        distances = dtw_distances(data)
    else:
        raise ValueError(
            "Unsupported distance metric. Use 'euclidean', 'manhattan', 'cosine', or 'dtw'."
        )

    print("Distances shape: ", distances.shape)

    try:
        assert distances.shape[0] == distances.shape[1]
    except AssertionError:
        print("Distances shape is not square")
        return
    try:
        assert distances.shape[0] == data.shape[0]
    except AssertionError:
        print("Distances shape is not equal to data shape")
        return

    # Create a graph
    G = nx.Graph()

    # Edge creation based on the chosen method
    G.add_nodes_from(range(len(data)))

    if connectivity == "epsilon":
        if clustering_algorithm in ["louvain", "spectral"]:
            percentile = 100
        print(f"Creating epsilon-neighborhood graph with percentile {percentile}...")
        epsilon = np.percentile(distances, percentile)
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if distances[i][j] < epsilon:
                    G.add_edge(i, j, weight=1 / distances[i][j])

    elif connectivity == "knn":
        print(f"Creating k-NN graph with {k_neighbors} neighbors...")
        knn_graph = kneighbors_graph(
            distances,  # Assuming  here is the appropriate input for kneighbors_graph
            n_neighbors=k_neighbors,
            metric="precomputed",
            mode="distance",
            include_self=False,
        )
        for i, j in zip(*knn_graph.nonzero()):
            G.add_edge(i, j, weight=1 / distances[i][j])
    # Check if the graph has edges
    try:
        assert len(G.edges) > 0
    except AssertionError:
        print("Graph has no edges")
        return
    print("# of edges: ", len(G.edges))
    print("# of nodes: ", len(G.nodes))

    # Store the experiment data
    exp_id = (
        str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        .replace(" ", "")
        .replace(":", "")
    )
    save_dir = f"./experiment_embedded_data/experiment_embedded_data_{exp_id}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    info_pd = pd.DataFrame(columns=["Item", "Value"])
    info_pd.loc[0] = ["Graph Construction Method", graph_construction_method]
    info_pd.loc[1] = ["Distance Metric", distance_metric]
    info_pd.loc[2] = ["Connectivity", connectivity]
    info_pd.loc[3] = ["K Neighbors", k_neighbors]
    info_pd.loc[4] = ["Percentile", percentile]
    info_pd.loc[5] = ["Clustering Algorithm", clustering_algorithm]
    info_pd.loc[6] = ["Model Name", model_name]

    if clustering_algorithm == "louvain":
        print(f"Running Louvain Algorithm...")
        partition = community_louvain.best_partition(G)
        # plot the community graph
        color_map = [partition.get(node) for node in G.nodes]
        plt.figure(figsize=(25, 25))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(
            G, pos, node_color=color_map, cmap=plt.cm.viridis, node_size=50
        )
        nx.draw_networkx_edges(G, pos, alpha=0.1)
        labels = {i: f"C{partition[i]}" for i in partition}
        nx.draw_networkx_labels(G, pos, labels, font_size=12)
        plt.title("Community Graph")
        plt.savefig(f"{save_dir}/community_graph.png")
        ## Graph Details
        nx.write_gexf(G, f"{save_dir}/graph.gexf")
        np.savetxt(
            f"{save_dir}/partition.csv",
            np.array(list(partition.values())),
            delimiter=",",
        )
        # Evaluate the partition
        info_pd.loc[7] = ["Modularity", evaluate_partition(G, partition)]
        info_pd.loc[8] = [
            "Silhouette Score",
            evaluate_partition_sil(data, partition),
        ]
        info_pd.loc[9] = ["Davies Bouldin Index", evaluate_davies_dbi(G, partition)]
        info_pd.loc[10] = ["Num Communities", len(set(partition.values()))]
        info_pd.to_csv(f"{save_dir}/info.csv", index=False)

    elif clustering_algorithm == "girvan_newman":
        plt.figure(figsize=(20, 20))
        print(f"Running Girvan Newman Algorithm...")

        if girvan_num_communities:
            print(f"Target Number of Communities: {girvan_num_communities}")
            communities, modularity = girvan_newman(
                G, num_communities=girvan_num_communities
            )
        elif girvan_target_modularity:
            print(f"Target Modularity: {girvan_target_modularity}")
            communities, modularity = girvan_newman(
                G, target_modularity=girvan_target_modularity
            )
        info_pd.loc[7] = ["girvan_modularity", girvan_target_modularity]
        info_pd.loc[8] = ["girvan_communities", girvan_num_communities]
        # plot the community graph
        pos = nx.spring_layout(G)

        partition = {}
        for cluster_index, nodes in enumerate(communities):
            for node in nodes:
                partition[node] = cluster_index
        color_map = [partition.get(node) for node in G.nodes]
        nx.draw_networkx(G, pos, node_color=color_map, with_labels=True, node_size=50)
        plt.title(
            f"Communities after Girvan-Newman Algorithm, Modularity: {modularity:.2f}"
        )
        plt.savefig(f"{save_dir}/community_graph.png")
        ## Graph Details
        nx.write_gexf(G, f"{save_dir}/graph.gexf")
        np.savetxt(f"{save_dir}/partition.csv", np.array(color_map), delimiter=",")
        # Evaluate the partition
        info_pd.loc[9] = ["Modularity", modularity]
        info_pd.loc[10] = ["Num Communities", len(communities)]
        info_pd.loc[11] = [
            "Silhouette Score",
            evaluate_partition_sil(data, partition),
        ]
        try:
            info_pd.loc[12] = [
                "Davies Bouldin Index",
                evaluate_davies_dbi(G, partition),
            ]
        except:
            info_pd.loc[12] = ["Davies Bouldin Index", "Error in calculation"]
        info_pd.to_csv(f"{save_dir}/info.csv", index=False)
    elif clustering_algorithm == "spectral":
        print(f"Running Spectral Clustering Algorithm...")
        L = nx.laplacian_matrix(G).astype(
            float
        )  # Ensure it's in float format for eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
        k_smallest_eigenvectors = eigenvectors[
            :, 1 : spectral_num_clusters + 1
        ]  # Skip the first eigenvector (constant)
        kmeans = KMeans(n_clusters=spectral_num_clusters)
        kmeans.fit(k_smallest_eigenvectors)
        clusters = kmeans.labels_
        partition = {i: clusters[i] for i in range(len(clusters))}
        try:
            assert len(set(partition.values())) == spectral_num_clusters
        except AssertionError:
            print("Number of clusters does not match the target")
            return
        try:
            print(calculate_modularity(G, clusters))
        except:
            print("Error in modularity calculation")
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G)
        color_map = [partition.get(node) for node in G.nodes]
        nx.draw_networkx(G, pos, node_color=color_map, with_labels=True, node_size=50)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.savefig(f"{save_dir}/Spectral_clustering_graph.png")
        ## Graph Details
        nx.write_gexf(G, f"{save_dir}/graph.gexf")
        np.savetxt(
            f"{save_dir}/partition.csv",
            np.array(list(partition.values())),
            delimiter=",",
        )
        # Evaluate the partition
        info_pd.loc[6] = ["Modularity", calculate_modularity(G, clusters)]
        info_pd.loc[7] = [
            "Silhouette Score",
            evaluate_partition_sil(real_data, partition),
        ]
        info_pd.loc[8] = ["Davies Bouldin Index", evaluate_davies_dbi(G, partition)]
        info_pd.loc[9] = ["Num Communities", len(set(partition.values()))]
        info_pd.to_csv(f"{save_dir}/info.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run community detection on generated data with various configurations."
    )

    # Graph construction method
    parser.add_argument(
        "--graph_construction_method",
        type=str,
        default="flatten",
        choices=[
            "flatten"
        ],  # Note: aggregate is not supported in this method - output of the model is 1-D
        help="""
        The method to construct the graph. Default is flatten. In case the distance set to dtw, this parameter is ignored.
        - flatten: Use the original data as is and flattent the data into the shape of (samples , timestamps * features)
        - aggregate: Aggregate the data by taking the mean of each feature over the timestamps. The shape of the data will be (samples, aggregated_features)
        in case of dtw, the distance metric can handle the original data shape. Therefore, this parameter is ignored.
        
        """,
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan", "cosine", "dtw"],
        help="The distance metric to use (euclidean, manhattan, cosine, dtw). Default is euclidean.",
    )
    parser.add_argument(
        "--connectivity",
        type=str,
        default="epsilon",
        choices=["epsilon", "knn"],
        help="The method to create graph edges (epsilon for epsilon-neighborhood, knn for k-nearest neighbors). Default is knn.",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=5,
        help="Number of neighbors to use for k-NN method. Default is 5.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=20,
        help="""
        In case of epsilon-neighborhood method, the percentile to use for epsilon. Default is 20.
        """,
    )

    # Data source
    parser.add_argument(
        "--data_source",
        type=str,
        default="29var",
        choices=["29var", "12var"],
        help="The data source to use. Default is 29var.",
    )
    parser.add_argument(
        "--break_data",
        type=bool,
        default=False,
        help="Break the data into smaller chunks. Default is False.",
    )
    parser.add_argument(
        "--break_size",
        type=int,
        default=100,
        help="Size of the smaller chunks. Default is 100.",
    )
    parser.add_argument(
        "--leave_out_problematic_features",
        type=bool,
        default=False,
        help="Leave out problematic features. Default is False.",
    )
    parser.add_argument(
        "--cutoff_data",
        type=bool,
        default=True,
        help="Cut off the data at a certain point. Default is 0.",
    )
    # Clustering Algorithm
    parser.add_argument(
        "--clustering_algorithm",
        type=str,
        default="louvain",
        choices=[
            "louvain",
            "spectral",
            "girvan_newman",
            "walktrap",
            "leading_eigenvector",
            "fast_greedy",
            "infomap",
            "label_propagation",
            "multilevel",
            "spinglass",
        ],
        help="The clustering algorithm to use. Default is louvain.",
    )
    parser.add_argument(
        "--girvan_num_communities",
        default=None,
        help="Number of communities to find with Girvan-Newman algorithm. Default is None.",
    )
    parser.add_argument(
        "--girvan_target_modularity",
        default=0.1,
        help="Target modularity to reach with Girvan-Newman algorithm. Default is None.",
    )
    parser.add_argument(
        "--spectral_num_clusters",
        type=int,
        default=5,
        help="Number of clusters to find with Spectral Clustering algorithm. Default is 5.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="TimeVAE_model9",
        help="Name of the model used to generate the data. Default is TimeVAE_model8.",
    )
    args = parser.parse_args()

    main(
        args.distance,
        args.connectivity,
        args.k_neighbors,
        args.graph_construction_method,
        args.percentile,
        args.clustering_algorithm,
        args.girvan_num_communities,
        args.girvan_target_modularity,
        args.spectral_num_clusters,
        args.model_name,
    )
