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
    save_dir = f"./experiment_Spectral_Embedded/experiment_{exp_id}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


from scipy.stats import entropy

def calculate_entropy(latent_vectors, labels):
    num_elements = len(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / num_elements
    return entropy(probabilities)


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
    coverage = covered_edges / total_edges if total_edges > 0 else 0
    return coverage


def calculate_silhouette(latent_vectors, labels, distance_matrix):
    # Number of elements (rows or columns)
    n_elements = latent_vectors.shape[0]

    # Initialize arrays to store silhouette coefficients and cluster indices
    silhouette_vals = np.zeros(n_elements)
    cluster_indices = np.unique(labels)

    # Compute silhouette coefficient for each element
    for idx, element in enumerate(latent_vectors):
        element_cluster = labels[idx]

        # Compute average dissimilarity within the same cluster (a(i))
        a_i = np.mean(distance_matrix[idx, labels == element_cluster])

        # Compute average dissimilarity to elements in the nearest neighboring cluster (b(i))
        b_i = np.min(
            [
                np.mean(distance_matrix[idx, labels == other_cluster])
                for other_cluster in cluster_indices
                if other_cluster != element_cluster
            ]
        )

        # Calculate silhouette coefficient for the element
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
    distance_metric,
    n_clusters,
    model_name,
):
    print("Latent Vectors Shape:", latent_vectors.shape)
    print(f"Distance Metric: {distance_metric}")
    print(f"Number of Clusters: {n_clusters}")

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
    distance_matrix_samples = pairwise_distances(latent_vectors, metric=distance_metric)
    distance_matrix_features = pairwise_distances(
        latent_vectors.T, metric=distance_metric
    )
    coverage = calculate_coverage(latent_vectors, row_labels, column_labels)
    entropy_samples = calculate_entropy(latent_vectors, row_labels)
    entropy_features = calculate_entropy(latent_vectors.T, column_labels)

    silhouette_samples_val = calculate_silhouette(
        latent_vectors, row_labels, distance_matrix_samples
    )
    silhouette_features_val = calculate_silhouette(
        latent_vectors.T, column_labels, distance_matrix_features
    )
    modularity = calculate_modularity(latent_vectors, row_labels, column_labels)
    metrics_dict = {
        "Method": "SpectralBiclustering",
        "Model_name": model_name,
        "Distance Metric": distance_metric,
        "Number of Clusters": n_clusters,
        "Coverage": coverage,
        "Entropy Samples": entropy_samples,
        "Entropy Features": entropy_features,
        "Silhouette Samples": silhouette_samples_val,
        "Silhouette Features": silhouette_features_val,
        "Modularity": modularity,
    }
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index", columns=["Value"])

    metrics_df.to_csv(f"{save_dir}/metrics.csv")

    # Save the results
    with open(f"{save_dir}/assignments.txt", "w") as f:
        for cluster_idx, cluster in row_clusters.items():
            for sample in cluster:
                f.write(f"sample {sample} cluster {cluster_idx}\n")

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
        default="TimeVAE_model9",
        help="Name of the model used to generate the data. Default is TimeGAN_model1.",
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
        args.distance_metric,
        args.n_clusters,
        args.model_name,
    )
