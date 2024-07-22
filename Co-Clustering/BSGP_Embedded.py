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
from itertools import combinations
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def create_dir():
    exp_id = (
        str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        .replace(" ", "")
        .replace(":", "")
    )
    save_dir = f"./experiment_BSGP_Embedded/experiment_{exp_id}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def calculate_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log(probabilities))


def calculate_coverage(sample_label, feature_label, data, num_clusters):
    covered_elements = 0
    for cluster_id in range(num_clusters):
        samples_in_cluster = np.where(sample_label == cluster_id)[0]
        features_in_cluster = np.where(feature_label == cluster_id)[0]
        covered_elements += len(samples_in_cluster) * len(features_in_cluster)
    total_elements = data.size
    return covered_elements / total_elements


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


def construct_bipartite_graph(data_matrix, threshold):
    B = nx.Graph()
    rows, cols = data_matrix.shape

    # Add row nodes
    B.add_nodes_from(range(rows), bipartite=0)

    # Add column nodes (shifted by row count to avoid overlap)
    B.add_nodes_from(range(rows, rows + cols), bipartite=1)

    # Add edges with weights based on the threshold
    for i in range(rows):
        for j in range(cols):
            if data_matrix[i, j] > threshold:  # Define a threshold for edge creation
                B.add_edge(i, rows + j, weight=data_matrix[i, j])

    return B


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


def bipartite_spectral_clustering(A, num_clusters):
    num_samples, num_features = A.shape

    # Degree matrices
    degrees_samples = np.sum(A, axis=1)
    degrees_features = np.sum(A, axis=0)

    D_samples = np.diag(degrees_samples)
    D_features = np.diag(degrees_features)

    # Construct the normalized Laplacian for the bipartite graph
    D_inv_sqrt_samples = np.diag(1.0 / np.sqrt(degrees_samples))
    D_inv_sqrt_features = np.diag(1.0 / np.sqrt(degrees_features))

    A_normalized = D_inv_sqrt_samples @ A @ D_inv_sqrt_features

    # Form the bipartite Laplacian
    B = np.block(
        [
            [np.zeros((num_samples, num_samples)), A_normalized],
            [A_normalized.T, np.zeros((num_features, num_features))],
        ]
    )

    # Compute the eigenvalues and eigenvectors
    eigvals, eigvecs = eigsh(B, k=num_clusters + 1, which="SM")

    # Skip the first eigenvector (corresponding to the smallest eigenvalue)
    eigvecs = eigvecs[:, 1 : num_clusters + 1]

    # Apply k-means clustering to the eigenvectors
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(eigvecs)
    labels = kmeans.labels_

    sample_label = labels[:num_samples]
    feature_label = labels[num_samples:]

    return sample_label, feature_label


def main(
    latent_vectors,
    aggregated_features,
    aggregated_features_names,
    percentile,
    num_clusters,
    model_name,
):
    print("Latent Vectors Shape:", latent_vectors.shape)
    print(f"percentile : {percentile}")
    print(f"model_name: {model_name}")
    print(f"num_clusters: {num_clusters}")

    # Perform bipartite spectral clustering
    sample_label, feature_label = bipartite_spectral_clustering(
        latent_vectors, num_clusters
    )
    print("Doc Labels:", sample_label)
    print("Word Labels:", feature_label)

    coverage = calculate_coverage(sample_label, feature_label, latent_vectors, num_clusters)
    entropy_samples = calculate_entropy(sample_label)
    entropy_features = calculate_entropy(feature_label)
    # modularity = calculate_modularity(latent_vectors, sample_label, feature_label)
    print(f"Coverage: {coverage}")
    print(f"Entropy (Samples): {entropy_samples}")
    print(f"Entropy (Features): {entropy_features}")
    # print(f"Modularity: {modularity}")

    save_dir = create_dir()
    # Write the sample cluster assignments to assignment.txt
    with open(f"{save_dir}/assignment.txt", "w") as f:
        for i, label in enumerate(sample_label):
            f.write(f"sample {i + 1} cluster {label + 1}\n")

    # Write the clusters to clusters.txt
    with open(f"{save_dir}/clusters.txt", "w") as f:
        for cluster_id in range(num_clusters):
            doc_cluster = np.where(sample_label == cluster_id)[0]
            word_cluster = np.where(feature_label == cluster_id)[0]
            f.write(
                f'Cluster {cluster_id + 1}: [{", ".join(map(str, doc_cluster + 1))}]\n'
            )
            f.write(
                f'Cluster {cluster_id + 1}: [{", ".join(map(str, word_cluster + 1))}]\n'
            )

    # Write the results to results.txt
    with open(f"{save_dir}/results.txt", "w") as f:
        for cluster_id in range(num_clusters):
            doc_cluster = np.where(sample_label == cluster_id)[0]
            word_cluster = np.where(feature_label == cluster_id)[0]
            f.write(
                f'Individual Cluster {cluster_id + 1} [{", ".join(map(str, doc_cluster + 1))}]\n'
            )
            f.write(
                f'Feature Cluster {cluster_id + 1} [{", ".join(map(str, word_cluster + 1))}]\n'
            )

    # Calculate silhouette scores for samples and features separately
    silhouette_samples = silhouette_score(latent_vectors, sample_label)
    silhouette_features = silhouette_score(latent_vectors.T, feature_label)

    # Calculate modularity
    threshold = np.percentile(latent_vectors, 75)
    B = construct_bipartite_graph(latent_vectors, threshold)
    communities = [
        np.where(sample_label == i)[0].tolist()
        + (np.where(feature_label == i)[0] + latent_vectors.shape[0]).tolist()
        for i in range(num_clusters)
    ]
    modularity = nx.algorithms.community.quality.modularity(B, communities)

    metrics = {
        "Method":"Bipartite Spectral Graph Partitioning",
        "Model Name": model_name,
        "Number of Clusters": num_clusters,
        "Coverage": coverage,
        "Entropy Samples": entropy_samples,
        "Entropy Features": entropy_features,
        "Silhouette Samples": silhouette_samples,
        "Silhouette Features": silhouette_features,
        "Modularity": modularity,
    }
    df_metrics = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    df_metrics.to_csv(f"{save_dir}/metrics.csv")
    # ### Interpretability
    # latent_feature_names = [f"Feature {i+1}" for i in range(len(latent_vectors[1]))]
    # correlation_matrix = np.zeros(
    #     (latent_vectors.shape[1], aggregated_features.shape[1])
    # )

    # for i in range(latent_vectors.shape[1]):
    #     for j in range(aggregated_features.shape[1]):
    #         correlation_matrix[i, j] = np.corrcoef(
    #             latent_vectors[:, i], aggregated_features[:, j]
    #         )[0, 1]

    # fig, ax = plt.subplots(figsize=(30, 30))
    # cax = ax.matshow(correlation_matrix, cmap="coolwarm")
    # fig.colorbar(cax)

    # # Setting the ticks for both axes
    # ax.set_xticks(np.arange(len(aggregated_features_names)))
    # ax.set_yticks(np.arange(len(latent_feature_names)))

    # # Labeling the ticks
    # ax.set_xticklabels(aggregated_features_names, rotation=90, fontsize=10)
    # ax.set_yticklabels(latent_feature_names, fontsize=10)

    # # Setting labels for x and y axis
    # ax.set_xlabel("Aggregated Features")
    # ax.set_ylabel("Latent Features")

    # # Improve layout to accommodate label sizes (especially useful if using rotation)
    # plt.tight_layout()
    # plt.title("Correlation Matrix for Latent Vectors vs Aggregated Features")
    # plt.savefig(f"{save_dir}/correlation_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SAMBA on generated data with various configurations."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="TimeVAE_model9",
        help="Name of the model used to generate the data. Default is TimeGAN_model1.",
    )
    parser.add_argument(
        "--percentile",
        type=str,
        default=75,
        help="Distance metric to use for clustering. Default is euclidean.",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=5,
        help="Number of clusters to use for clustering. Default is 5.",
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
        args.percentile,
        args.num_clusters,
        args.model_name,
    )
