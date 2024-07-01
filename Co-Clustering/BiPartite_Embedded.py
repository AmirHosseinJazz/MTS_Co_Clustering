import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import community
import os
from datetime import datetime
import argparse
from util import load_data
import ast
from scipy.stats import skew, kurtosis
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import pairwise_distances
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralBiclustering


def create_dir():
    exp_id = (
        str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        .replace(" ", "")
        .replace(":", "")
    )
    save_dir = f"./experiment_BiPartite_Embedded/experiment_{exp_id}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def calculate_entropy(latent_vectors, row_labels):
    num_samples = latent_vectors.shape[0]
    num_biclusters = len(np.unique(row_labels))

    # Initialize array to store within-cluster distances
    within_cluster_distances = np.zeros(num_biclusters)

    # Calculate centroids (mean vectors) for each bicluster
    for label in np.unique(row_labels):
        samples_in_cluster = latent_vectors[row_labels == label]
        centroid = np.mean(samples_in_cluster, axis=0)

        # Calculate squared Euclidean distance of each sample to the centroid
        distances = np.sum((samples_in_cluster - centroid) ** 2, axis=1)

        # Sum of squared distances within the bicluster
        within_cluster_distances[label] = np.sum(distances)

    # Calculate average within-cluster distance (entropy)
    entropy = np.mean(within_cluster_distances)

    return entropy


def calculate_coverage(latent_vectors, row_labels, column_labels):
    # Determine the number of unique clusters
    num_row_clusters = len(np.unique(row_labels))
    num_column_clusters = len(np.unique(column_labels))

    # Initialize the boolean matrix
    biclusters_ = np.zeros((num_row_clusters, num_column_clusters), dtype=bool)

    # Fill the boolean matrix based on row and column labels
    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(column_labels):
            biclusters_[row_label, col_label] = True

    total_edges = 0
    covered_edges = 0

    # Iterate through each row (individual) and check the bicluster labels
    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(column_labels):
            if biclusters_[row_label, col_label]:
                total_edges += 1
                if latent_vectors[row_idx, col_idx] != 0:
                    covered_edges += 1

    # Calculate coverage
    # print(total_edges, covered_edges)
    coverage = covered_edges / total_edges if total_edges > 0 else 0
    return coverage
    # print(f"Coverage: {coverage}")


def calculate_silhouette(latent_vectors, row_labels, distance_matrix):
    # Number of samples (rows)
    n_samples = latent_vectors.shape[0]

    # Initialize arrays to store silhouette coefficients and cluster indices
    silhouette_vals = np.zeros(n_samples)
    cluster_indices = np.unique(row_labels)

    # Compute silhouette coefficient for each sample
    for idx, sample in enumerate(latent_vectors):
        sample_cluster = row_labels[idx]

        # Compute average dissimilarity within the same cluster (a(i))
        a_i = np.mean(distance_matrix[idx, row_labels == sample_cluster])

        # Compute average dissimilarity to samples in the nearest neighboring cluster (b(i))
        b_i = np.min(
            [
                np.mean(distance_matrix[idx, row_labels == other_cluster])
                for other_cluster in cluster_indices
                if other_cluster != sample_cluster
            ]
        )

        # Calculate silhouette coefficient for the sample
        silhouette_vals[idx] = (b_i - a_i) / max(a_i, b_i)

    # Calculate average silhouette score
    average_silhouette = np.mean(silhouette_vals)
    return average_silhouette


def calculate_modularity(latent_vectors, row_labels, col_labels):
    # Construct adjacency matrix (example: based on non-zero entries in latent_vectors)
    adjacency_matrix = np.zeros_like(latent_vectors)
    adjacency_matrix[latent_vectors != 0] = 1  # Binary adjacency matrix

    # Get number of samples and features
    n_samples, n_features = latent_vectors.shape

    # Calculate modularity
    total_edges = np.sum(adjacency_matrix)
    degrees_samples = np.sum(adjacency_matrix, axis=1)
    degrees_features = np.sum(adjacency_matrix, axis=0)

    modularity = 0.0

    for i in range(n_samples):
        for j in range(n_features):
            if row_labels[i] == col_labels[j]:  # Check if same bicluster
                modularity += adjacency_matrix[i, j] - degrees_samples[
                    i
                ] * degrees_features[j] / (2 * total_edges)

    modularity /= 2 * total_edges
    return modularity


def greedy_modularity_calculate_modularity(B, partition):
    m = B.number_of_edges()
    Q = 0.0
    for u in B:
        for v in B:
            if partition[u] == partition[v] and B.has_edge(u, v):
                k_u = B.degree(u)
                k_v = B.degree(v)
                Q += 1 - (k_u * k_v) / (2 * m)
    return Q / (2 * m)


def main(
    latent_vectors,
    aggregated_features,
    aggregated_features_names,
    method,
    distance_metric,
    n_clusters,
):
    print("Latent Vectors Shape:", latent_vectors.shape)
    print(f"Bi-Clustering : {method}")
    print(f"Distance Metric: {distance_metric}")
    print(f"Number of Clusters: {n_clusters}")

    if method == "spectral":
        model = SpectralBiclustering(n_clusters=n_clusters).fit(latent_vectors)
        row_labels = model.row_labels_
        column_labels = model.column_labels_
        row_clusters = {}
        column_clusters = {}
        for sample_idx, label in enumerate(row_labels):
            if label not in row_clusters:
                row_clusters[label] = []
            row_clusters[label].append(
                f"{sample_idx + 1}"
            )  # Adjust sample numbering as needed

        for feature_idx, label in enumerate(column_labels):
            if label not in column_clusters:
                column_clusters[label] = []
            column_clusters[label].append(f"{feature_idx + 1}")

        save_dir = create_dir()
        with open(f"{save_dir}/results.txt", "w") as f:
            for cluster_idx, cluster in row_clusters.items():
                f.write(f"Individual Cluster {cluster_idx}: {cluster}\n")
            for cluster_idx, cluster in column_clusters.items():
                f.write(f"Feature Cluster {cluster_idx}: {cluster}\n")

        with open(f"{save_dir}/clusters.txt", "w") as f:
            for cluster_idx, cluster in row_clusters.items():
                f.write(f"Cluster {cluster_idx + 1}: {cluster}\n")
            for cluster_idx, cluster in column_clusters.items():
                f.write(f"Cluster {cluster_idx + 1}: {cluster}\n")

        ### Metrics
        distance_matrix = pairwise_distances(latent_vectors, metric=distance_metric)
        coverage = calculate_coverage(latent_vectors, row_labels, column_labels)
        entropy = calculate_entropy(latent_vectors, row_labels)

        silhouette = calculate_silhouette(latent_vectors, row_labels, distance_matrix)
        modularity = calculate_modularity(latent_vectors, row_labels, column_labels)
        metrics_dict = {
            "Method": method,
            "Distance Metric": distance_metric,
            "Number of Clusters": n_clusters,
            "Coverage": coverage,
            "Entropy": entropy,
            "Silhouette": silhouette,
            "Modularity": modularity,
        }
        metrics_df = pd.DataFrame.from_dict(
            metrics_dict, orient="index", columns=["Value"]
        )

        metrics_df.to_csv(f"{save_dir}/metrics.csv")

        # Save the results
        with open(f"{save_dir}/assignments.txt", "w") as f:
            for cluster_idx, cluster in row_clusters.items():
                for sample in cluster:
                    f.write(f"sample {sample} cluster {cluster_idx}\n")

    elif method == "greedy_modularity":
        B = nx.Graph()
        # Add nodes with the bipartite attribute
        individuals = range(latent_vectors.shape[0])
        features = range(
            latent_vectors.shape[0], latent_vectors.shape[0] + latent_vectors.shape[1]
        )
        B.add_nodes_from(individuals, bipartite=0)
        B.add_nodes_from(features, bipartite=1)

        # Add edges with weights based on the latent vector values
        for i in individuals:
            for j, feature in enumerate(features):
                weight = latent_vectors[i, j]
                B.add_edge(i, feature, weight=weight)

        # Perform bipartite community detection using a bipartite algorithm
        communities = community.greedy_modularity_communities(B, weight="weight")

        # Calculate modularity
        # modularity = greedy_modularity_calculate_modularity(B, communities)
        # Calculate Coverage
        total_edges = B.number_of_edges()
        covered_edges = sum(
            len(set(B.neighbors(node)) & set(B.neighbors(neighbor)))
            for _community in communities
            for node in _community
            for neighbor in _community
        )
        coverage = covered_edges / total_edges

        # Caclulate Entropy
        total_nodes = sum(len(c) for c in communities)
        entropy = 0.0
        for _community in communities:
            if _community:
                # Calculate probability of each community
                p = len(_community) / total_nodes
                if p > 0:
                    entropy -= p * np.log2(p)  # Use log2 for entropy in bits

        # # Calculate Silhouette Score
        labels = np.zeros(latent_vectors.shape[0])

        # Assign cluster labels based on communities
        for idx, comm in enumerate(communities):
            for node in comm:
                if node < latent_vectors.shape[0]:  # Check if node is an individual
                    labels[node] = idx

        # Calculate silhouette scores for individuals
        silhouette = silhouette_samples(
            latent_vectors[: latent_vectors.shape[0]], labels, metric="euclidean"
        )
        average_silhouette = np.mean(silhouette)

        metrics_dict = {
            "Method": method,
            "Modularity": modularity,
            "Coverage": coverage,
            "Entropy": entropy,
            "Silhouette": average_silhouette,
        }
        metrics_df = pd.DataFrame.from_dict(
            metrics_dict, orient="index", columns=["Value"]
        )
        # Save the results

        save_dir = create_dir()

        vertical_spacing = (
            10  # change the spacing value as needed for better visibility
        )
        pos = {}

        for index, individual in enumerate(individuals):
            pos[individual] = (
                -100,
                index * vertical_spacing,
            )  # Column 0 for individuals

        for index, feature in enumerate(features):
            pos[feature] = (100, index * vertical_spacing)  # Column 1 for features

        node_colors = [
            "blue" if B.nodes[node]["bipartite"] == 0 else "green" for node in B
        ]

        plt.figure(figsize=(50, 50))
        nx.draw(
            B,
            pos,
            node_color=node_colors,
            node_size=50,
            with_labels=True,
            cmap=plt.cm.jet,
        )
        plt.title("Community Detection in Bipartite Graph")

        metrics_df.to_csv(f"{save_dir}/metrics.csv")

        plt.savefig(f"{save_dir}/community_co_clustering.png")

        community_nodes = [list(com) for com in communities]
        individual_clusters = [
            list(filter(lambda x: x in individuals, com)) for com in communities
        ]
        feature_clusters = [
            list(filter(lambda x: x in features, com)) for com in communities
        ]

        # Save the results
        with open(f"{save_dir}/results.txt", "w") as f:
            for cluster in individual_clusters:
                f.write(f"Individual Cluster: {cluster}\n")
                f.write("\n")
            for cluster in feature_clusters:
                Item = []
                for item in cluster:
                    converted_item = int(item)
                    multiplied_item = converted_item - latent_vectors.shape[0]
                    Item.append(multiplied_item)
                f.write(f"Feature Cluster: {Item}\n")

        with open(f"{save_dir}/assignments.txt", "w") as f:
            for cluster_idx, community_nodes in enumerate(communities):
                for node in community_nodes:
                    if node in individuals:
                        f.write(f"sample {node} cluster {cluster_idx}\n")

        with open(f"{save_dir}/cluster.txt", "w") as f:
            for cluster_idx, community_nodes in enumerate(communities):
                for node in community_nodes:
                    if node in individuals:
                        f.write(f"Cluster {cluster_idx + 1}: Sample {node}\n")
                    else:
                        f.write(
                            f"Cluster {cluster_idx + 1}: Feature {node - latent_vectors.shape[0]}\n"
                        )

    elif method == "BiMCL":
        # Initialize bipartite graph B
        B = nx.Graph()

        # Add nodes with bipartite attribute
        individuals = range(latent_vectors.shape[0])
        features = range(
            latent_vectors.shape[0], latent_vectors.shape[0] + latent_vectors.shape[1]
        )
        B.add_nodes_from(individuals, bipartite=0)
        B.add_nodes_from(features, bipartite=1)

        # Add edges with weights based on latent vector values
        for i in individuals:
            for j, feature in enumerate(features):
                weight = latent_vectors[i, j]
                B.add_edge(i, feature, weight=weight)

        # Define parameters
        max_iterations = 100  # Maximum number of iterations

        # Initialize adjacency matrix or weight matrix
        adjacency_matrix = np.array(nx.to_numpy_array(B))
        print("Adjacency Matrix Shape:", adjacency_matrix.shape)
        # Normalize adjacency matrix
        normalized_matrix = adjacency_matrix / np.max(adjacency_matrix)

        # Initialize bicluster assignments randomly
        row_labels = np.random.randint(0, n_clusters, size=latent_vectors.shape[0])
        column_labels = np.random.randint(0, n_clusters, size=latent_vectors.shape[1])

        # Iterative BiMCL
        for iteration in range(max_iterations):
            # Calculate mean cut level between current biclusters
            mean_cut_level = np.zeros((n_clusters, n_clusters))

            # Iterate over each bicluster pair (r, c)
            for r in range(n_clusters):
                for c in range(n_clusters):
                    # Select rows and columns corresponding to bicluster r and c
                    rows_in_bicluster_r = np.where(row_labels == r)[
                        0
                    ]  # Get indices of rows in bicluster r
                    cols_in_bicluster_c = np.where(column_labels == c)[
                        0
                    ]  # Get indices of columns in bicluster c

                    # Calculate mean cut level between rows_in_bicluster_r and cols_in_bicluster_c
                    if len(rows_in_bicluster_r) > 0 and len(cols_in_bicluster_c) > 0:
                        mean_cut_level[r, c] = np.mean(
                            normalized_matrix[
                                rows_in_bicluster_r[:, None], cols_in_bicluster_c
                            ]
                        )
                    else:
                        mean_cut_level[r, c] = (
                            0.0  # Handle case where bicluster is empty
                        )

            # Update bicluster assignments based on mean cut level
            for i in range(latent_vectors.shape[0]):
                for j in range(latent_vectors.shape[1]):
                    current_row_label = row_labels[i]
                    current_column_label = column_labels[j]
                    best_row_label = current_row_label
                    best_column_label = current_column_label
                    best_score = mean_cut_level[current_row_label, current_column_label]

                    # Try reassigning rows and columns to other biclusters
                    for r in range(n_clusters):
                        for c in range(n_clusters):
                            if r != current_row_label or c != current_column_label:
                                new_score = mean_cut_level[r, c]
                                if new_score > best_score:
                                    best_score = new_score
                                    best_row_label = r
                                    best_column_label = c

                    # Update labels
                    row_labels[i] = best_row_label
                    column_labels[j] = best_column_label
        save_dir = create_dir()
        individual_clusters = {cluster_idx: [] for cluster_idx in np.unique(row_labels)}
        feature_clusters = {cluster_idx: [] for cluster_idx in np.unique(column_labels)}

        # Populate individual_clusters and feature_clusters
        for sample_idx, cluster_idx in enumerate(row_labels):
            individual_clusters[cluster_idx].append(f"individual {sample_idx + 1}")

        for feature_idx, cluster_idx in enumerate(column_labels):
            feature_clusters[cluster_idx].append(f"feature {feature_idx + 1}")

        # Write to assignments.txt
        with open(f"{save_dir}/assignments.txt", "w") as f:
            for sample_idx, cluster_idx in enumerate(row_labels):
                f.write(f"sample {sample_idx + 1}: cluster {cluster_idx}\n")

        # Write to one_cluster.txt
        with open(f"{save_dir}/clusters.txt", "w") as f:
            for cluster_idx, cluster in individual_clusters.items():
                f.write(f"Cluster {cluster_idx}: {', '.join(cluster)}\n")

            f.write("\n")

            for cluster_idx, cluster in feature_clusters.items():
                f.write(f"Cluster {cluster_idx}: {', '.join(cluster)}\n")

        # Write to result.txt
        with open(f"{save_dir}/result.txt", "w") as f:
            for cluster_idx, cluster in individual_clusters.items():
                f.write(f"Individual Cluster {cluster_idx}: {cluster}\n")

            f.write("\n")

            for cluster_idx, cluster in feature_clusters.items():
                f.write(f"Feature Cluster {cluster_idx}: {cluster}\n")

        # Calculate metrics
        distance_matrix = pairwise_distances(latent_vectors, metric=distance_metric)
        coverage = calculate_coverage(latent_vectors, row_labels, column_labels)
        entropy = calculate_entropy(latent_vectors, row_labels)
        silhouette = calculate_silhouette(latent_vectors, row_labels, distance_matrix)
        modularity = calculate_modularity(latent_vectors, row_labels, column_labels)

        metrics_dict = {
            "Method": method,
            "Distance Metric": distance_metric,
            "Number of Clusters": n_clusters,
            "Coverage": coverage,
            "Entropy": entropy,
            "Silhouette": silhouette,
            "Modularity": modularity,
        }
        metrics_df = pd.DataFrame.from_dict(
            metrics_dict, orient="index", columns=["Value"]
        )

        metrics_df.to_csv(f"{save_dir}/metrics.csv")

    ### Interpretability
    latent_feature_names = [f"Feature {i+1}" for i in range(len(latent_vectors[1]))]
    correlation_matrix = np.zeros(
        (latent_vectors.shape[1], aggregated_features.shape[1])
    )

    for i in range(latent_vectors.shape[1]):
        for j in range(aggregated_features.shape[1]):
            correlation_matrix[i, j] = np.corrcoef(
                latent_vectors[:, i], aggregated_features[:, j]
            )[0, 1]

    fig, ax = plt.subplots(figsize=(30, 30))
    cax = ax.matshow(correlation_matrix, cmap="coolwarm")
    fig.colorbar(cax)

    # Setting the ticks for both axes
    ax.set_xticks(np.arange(len(aggregated_features_names)))
    ax.set_yticks(np.arange(len(latent_feature_names)))

    # Labeling the ticks
    ax.set_xticklabels(aggregated_features_names, rotation=90, fontsize=10)
    ax.set_yticklabels(latent_feature_names, fontsize=10)

    # Setting labels for x and y axis
    ax.set_xlabel("Aggregated Features")
    ax.set_ylabel("Latent Features")

    # Improve layout to accommodate label sizes (especially useful if using rotation)
    plt.tight_layout()
    plt.title("Correlation Matrix for Latent Vectors vs Aggregated Features")
    plt.savefig(f"{save_dir}/correlation_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run community detection on generated data with various configurations."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="TimeVAE_model8",
        help="Name of the model used to generate the data. Default is TimeGAN_model1.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["spectral", "greedy_modularity", "BiMCL"],
        default="spectral",
        help="Method to use for biClustering. Default is greedy_modularity.",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        choices=[
            "euclidean",
            "cosine",
            "dtw",
            "manhattan",
        ],
        default="euclidean",
        help="Distance metric to use for clustering. Default is euclidean.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=3,
        help="Number of clusters to use for clustering. Default is 3.",
    )

    args = parser.parse_args()
    latent_vectors = np.load(f"../Generated/{args.model_name}/encoded_data.npy")
    print("Latent Vectors Shape:", latent_vectors.shape)

    params = {}
    if "GAN" in args.model_name:

        try:
            with open(f"../TGAN/saved_models/{args.model_name}/config.txt", "r") as f:
                for line in f:
                    key, value = line.split(":")
                    # check if the parameter is an integer
                    try:
                        params[key.strip()] = int(value.strip())
                    except:
                        params[key.strip()] = value.strip()
        except Exception as E:
            raise Exception(f"Could not load the configuration file: {E}")
    elif "VAE" in args.model_name:
        try:
            with open(f"../TVAE/saved_models/{args.model_name}/config.txt", "r") as f:
                for line in f:
                    key, value = line.split(":")
                    # check if the parameter is an integer
                    try:
                        params[key.strip()] = int(value.strip())
                    except:
                        params[key.strip()] = value.strip()
        except Exception as E:
            raise Exception(f"Could not load the configuration file: {E}")

    # Read model config from a text file
    model_config = {}
    config_file = (
        f"../TGAN/saved_models/{args.model_name}/config.txt"
        if "GAN" in args.model_name
        else f"../TVAE/saved_models/{args.model_name}/config.txt"
    )
    try:
        with open(config_file, "r") as f:
            for line in f:
                key, value = line.split(":")
                # check if the parameter is an integer
                try:
                    model_config[key.strip()] = int(value.strip())
                except:
                    model_config[key.strip()] = value.strip()
    except Exception as E:
        raise Exception(f"Could not load the configuration file: {E}")

    if model_config["data_source"] == "29var":
        real_data, feature_names = load_data(
            "../Data/PreProcessed/29var/df29.xlsx",
            break_to_smaller=ast.literal_eval(model_config["break_data"]),
            break_size=model_config["break_size"],
            leave_out_problematic_features=ast.literal_eval(
                model_config["leave_out_problematic_features"]
            ),
            cutoff_data=model_config["cutoff_data"],
            feature_shape=latent_vectors.shape[-1],
        )
    elif model_config["data_source"] == "12var":
        real_data, feature_names = load_data(
            "../Data/PreProcessed/12var/df12.xlsx",
            break_to_smaller=ast.literal_eval(model_config["break_data"]),
            break_size=model_config["break_size"],
            leave_out_problematic_features=ast.literal_eval(
                model_config["leave_out_problematic_features"]
            ),
            cutoff_data=model_config["cutoff_data"],
        )
    # print("Real Data Shape:", real_data.shape)
    # print("Feature Names:", feature_names)
    # print("Feature Names Shape:", len(feature_names))

    num_features = real_data.shape[2]
    num_statistics = 3  # mean, std, range, skew, kurtosis
    aggregated_features = np.empty((real_data.shape[0], num_features * num_statistics))

    for i in range(num_features):
        # Calculate mean and std over timestamps
        feature_data = real_data[:, :, i]
        feature_mean = np.mean(feature_data, axis=1)
        feature_std = np.std(feature_data, axis=1)
        feature_range = np.ptp(feature_data, axis=1)  # Peak to peak range (max-min)
        # feature_skew = skew(feature_data, axis=1)
        # feature_kurtosis = kurtosis(feature_data, axis=1)
        # Store calculated statistics in the corresponding columns
        index_base = i * num_statistics
        aggregated_features[:, index_base] = feature_mean
        aggregated_features[:, index_base + 1] = feature_std
        aggregated_features[:, index_base + 2] = feature_range
        # aggregated_features[:, index_base + 3] = feature_skew
        # aggregated_features[:, index_base + 4] = feature_kurtosis

    # Print the shape of the aggregated features to verify
    print("Aggregated Features Shape:", aggregated_features.shape)

    # Generate names for each aggregated feature
    aggregated_features_names = []
    for feature_name in feature_names:
        aggregated_features_names.extend(
            [
                f"{feature_name}_mean",
                f"{feature_name}_std",
                f"{feature_name}_range",
                # f"{feature_name}_skew",
                # f"{feature_name}_kurtosis",
            ]
        )
    # print("Aggregated Features Names:", aggregated_features_names)
    main(
        latent_vectors,
        aggregated_features,
        aggregated_features_names[:],
        args.method,
        args.distance_metric,
        args.n_clusters,
    )
