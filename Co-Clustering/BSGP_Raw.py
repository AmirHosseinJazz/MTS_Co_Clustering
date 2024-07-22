import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
import networkx as nx
import os
from datetime import datetime
import argparse
from collections import Counter
import math
import ast
from util import load_data


def create_dir():
    exp_id = (
        str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        .replace(" ", "")
        .replace(":", "")
    )
    save_dir = f"./experiment_BSGP_Raw/experiment_{exp_id}"
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
    data,
    distance_metric,
    percentile,
    n_clusters,
    aggregation,
    model_name,
    feature_names,
):
    print("Original Data Shape:", data.shape)

    # Apply the specified aggregation method
    if aggregation == "flatten":
        data_processed = data.reshape(data.shape[0], -1)
    elif aggregation == "mean":
        data_processed = np.mean(data, axis=1)
    else:
        raise ValueError("Unsupported aggregation method")

    print("Processed Data Shape:", data_processed.shape)

    # Normalize the data
    scaler = StandardScaler()
    data_processed = scaler.fit_transform(data_processed)

    # Perform bipartite spectral clustering
    sample_label, feature_label = bipartite_spectral_clustering(
        data_processed, n_clusters
    )
    print("Sample Labels:", sample_label)
    print("Feature Labels:", feature_label)

    coverage = calculate_coverage(
        sample_label, feature_label, data_processed, n_clusters
    )
    entropy_samples = calculate_entropy(sample_label)
    entropy_features = calculate_entropy(feature_label)
    print(f"Coverage: {coverage}")
    print(f"Entropy (Samples): {entropy_samples}")
    print(f"Entropy (Features): {entropy_features}")

    save_dir = create_dir()
    # Write the sample cluster assignments to assignment.txt
    with open(f"{save_dir}/assignment.txt", "w") as f:
        for i, label in enumerate(sample_label):
            f.write(f"sample {i + 1} cluster {label + 1}\n")

    # Write the clusters to clusters.txt
    with open(f"{save_dir}/clusters.txt", "w") as f:
        for cluster_id in range(n_clusters):
            sample_cluster = np.where(sample_label == cluster_id)[0]
            feature_cluster = np.where(feature_label == cluster_id)[0]
            f.write(
                f'Cluster {cluster_id + 1}: [{", ".join(map(str, sample_cluster + 1))}]\n'
            )
            f.write(
                f'Cluster {cluster_id + 1}: [{", ".join(map(str, feature_cluster + 1))}]\n'
            )

    # Write the results to results.txt
    with open(f"{save_dir}/results.txt", "w") as f:
        for cluster_id in range(n_clusters):
            sample_cluster = np.where(sample_label == cluster_id)[0]
            feature_cluster = np.where(feature_label == cluster_id)[0]
            f.write(
                f'Individual Cluster {cluster_id + 1} [{", ".join(map(str, sample_cluster + 1))}]\n'
            )
            f.write(
                f'Feature Cluster {cluster_id + 1} [{", ".join(map(str, feature_cluster + 1))}]\n'
            )

    # Calculate silhouette scores for samples and features separately
    distance_matrix_samples = pairwise_distances(data_processed, metric=distance_metric)
    distance_matrix_features = pairwise_distances(
        data_processed.T, metric=distance_metric
    )
    silhouette_samples_val = silhouette_score(
        distance_matrix_samples, sample_label, metric="precomputed"
    )
    silhouette_features_val = silhouette_score(
        distance_matrix_features, feature_label, metric="precomputed"
    )

    # Calculate modularity
    threshold = np.percentile(data_processed, int(percentile))
    B = construct_bipartite_graph(data_processed, threshold)
    communities = [
        np.where(sample_label == i)[0].tolist()
        + (np.where(feature_label == i)[0] + data_processed.shape[0]).tolist()
        for i in range(n_clusters)
    ]
    modularity = nx.algorithms.community.quality.modularity(B, communities)

    metrics = {
        "Method": "Bipartite Spectral Graph Partitioning",
        "Model Name": model_name,
        "Number of Clusters": n_clusters,
        "Coverage": coverage,
        "Entropy Samples": entropy_samples,
        "Entropy Features": entropy_features,
        "Silhouette Samples": silhouette_samples_val,
        "Silhouette Features": silhouette_features_val,
        "Modularity": modularity,
    }
    df_metrics = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    df_metrics.to_csv(f"{save_dir}/metrics.csv")


if __name__ == "__main__":
    # Parse command-line arguments

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
        default="euclidean",
        choices=["euclidean", "cosine", "manhattan", "dtw"],
        help="Distance metric to use for clustering. Default is euclidean.",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=3,
        help="Number of clusters to use for clustering. Default is 3.",
    )
    parser.add_argument(
        "--percentile",
        type=str,
        default=90,
        help="Percentile to use for thresholding. Default is 90.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="flatten",
        choices=["flatten", "mean"],
        help="Aggregation method to use for clustering. Default is flatten",
    )
    args = parser.parse_args()
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
    print("Real Data Shape:", real_data.shape)

    # Run the main function
    main(
        real_data,
        args.distance_metric,
        args.percentile,
        args.num_clusters,
        args.aggregation,
        args.model_name,
        feature_names,
    )
