import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import (
    euclidean_distances,
    manhattan_distances,
    cosine_distances,
)
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score
from fastdtw import fastdtw
from sklearn.metrics import pairwise_distances

from scipy.spatial.distance import euclidean

import community as community_louvain
import argparse
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
from util import load_data
import ast
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
    silhouette_score,
)
import networkx as nx
from sklearn.cluster import SpectralClustering
import community as community_louvain
from sklearn.preprocessing import StandardScaler
from util import load_data
import argparse
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
    silhouette_score,
)


def individual_clustering(data, n_clusters, method, distance_metric):
    if distance_metric == "dtw":
        dist_func = lambda x, y: fastdtw(x, y)[0]
    else:
        dist_func = distance_metric

    if method == "spectral":
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity=dist_func,
            assign_labels="discretize",
            random_state=0,
        ).fit(data)
        labels = clustering.labels_
    elif method == "greedy_modularity":
        labels = list(community_louvain.best_partition(nx.Graph(data)).values())
    elif method == "louvain":
        labels = list(community_louvain.best_partition(nx.Graph(data)).values())

    return labels


def feature_clustering(data, n_clusters, method, distance_metric):
    if distance_metric == "dtw":
        dist_func = lambda x, y: fastdtw(x, y)[0]
    else:
        dist_func = distance_metric

    if method == "spectral":
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity=dist_func,
            assign_labels="discretize",
            random_state=0,
        ).fit(data.T)
        labels = clustering.labels_
    elif method == "greedy_comm":
        labels = list(community_louvain.best_partition(nx.Graph(data)).values())
    elif method == "louvain":
        labels = list(community_louvain.best_partition(nx.Graph(data)).values())

    return labels


def evaluate_clustering_metrics(data, labels, true_labels):
    metrics = {}
    metrics["Adjusted Rand Index"] = adjusted_rand_score(true_labels, labels)
    metrics["Adjusted Mutual Information"] = adjusted_mutual_info_score(
        true_labels, labels
    )
    metrics["Completeness Score"] = completeness_score(true_labels, labels)
    metrics["Homogeneity Score"] = homogeneity_score(true_labels, labels)
    metrics["V-measure"] = v_measure_score(true_labels, labels)
    # Calculate Silhouette Score if labels are not all the same
    if len(np.unique(labels)) > 1:
        metrics["Silhouette Score"] = silhouette_score(data, labels)
    else:
        metrics["Silhouette Score"] = 0.0

    return metrics


def coverage_score(true_labels, labels):
    # Calculate contingency matrix
    contingency = contingency_matrix(true_labels, labels)

    # Calculate coverage as the sum of the maximum values in each row divided by the total number of samples
    coverage = np.sum(np.max(contingency, axis=1)) / np.sum(contingency)

    return coverage


def main(
    normalize,
    data,
    feature_names,
    n_clusters_level1,
    n_clusters_level2,
    mode,
    method1,
    method2,
    distance_metric,
):
    # Normalize the data if specified
    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    if mode == "aggregated":
        data = np.mean(data, axis=1)
        print("Data Shape:", data.shape)

    else:
        print("Data Shape:", data.shape)

    if distance_metric == "dtw":
        distances = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                dist, _ = fastdtw(
                    data[i].reshape(-1, 1), data[j].reshape(-1, 1), dist=euclidean
                )
                distances[i, j] = dist

    else:
        distances = pairwise_distances(data, metric=distance_metric)
    gamma = 1.0
    affinity_matrix = np.exp(-gamma * distances**2)

    if method1 == "spectral":
        print("Affinity Matrix Shape:", affinity_matrix.shape)
        print("n_clusters_level1:", n_clusters_level1)
        clustering = SpectralClustering(
            n_clusters=n_clusters_level1,
            affinity="precomputed",
            random_state=0,
        ).fit(affinity_matrix)
        labels = clustering.labels_
    elif method1 == "greedy_modularity":

        labels = list(community_louvain.best_partition(nx.Graph(data)).values())
    elif method1 == "louvain":
        labels = list(community_louvain.best_partition(nx.Graph(data)).values())

    print(labels)

    # # Perform individual sample clustering
    # labels_level1 = individual_clustering(
    #     data, n_clusters_level1, method1, distance_metric
    # )

    # # Evaluate metrics for individual sample clustering
    # metrics_level1 = evaluate_clustering_metrics(data, labels_level1, true_labels=None)

    # # Save inputs of clusters at level 1
    # level1_clusters = {}
    # for cluster_id in range(n_clusters_level1):
    #     level1_clusters[cluster_id] = data[labels_level1 == cluster_id]

    # # Perform feature-level clustering within each cluster at level 1
    # level2_clusters = {}
    # for cluster_id, cluster_data in level1_clusters.items():
    #     labels_level2 = feature_clustering(
    #         cluster_data, n_clusters_level2, method2, distance_metric
    #     )
    #     level2_clusters[cluster_id] = labels_level2

    # # Save inputs of clusters at level 2
    # for cluster_id, feature_labels in level2_clusters.items():
    #     for sub_cluster_id in range(n_clusters_level2):
    #         cluster_data = level1_clusters[cluster_id][feature_labels == sub_cluster_id]
    #         # Save or process cluster_data as needed

    # # Print or save metrics and results
    # print("Level 1 Clustering Metrics:")
    # for metric, value in metrics_level1.items():
    #     print(f"{metric}: {value}")

    # # Example of saving results
    # if not os.path.exists("./clustering_results/"):
    #     os.makedirs("./clustering_results/")

    # # Save metrics and results
    # pd.DataFrame(metrics_level1.items(), columns=["Metric", "Value"]).to_csv(
    #     "./clustering_results/level1_metrics.csv", index=False
    # )


if __name__ == "__main__":
    # Parse command-line arguments

    parser = argparse.ArgumentParser(
        description="Run two-level community detection on generated data with various configurations."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the input data. Default is False.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="TimeVAE_model8",
        help="Name of the model used to generate the data. Default is TimeGAN_model1.",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="euclidean",
        help="Distance metric to use for clustering. Default is euclidean.",
        choices=["euclidean", "cosine", "manhattan", "dtw"],
    )
    parser.add_argument(
        "--n_clusters_level1",
        type=int,
        default=10,
        help="Number of clusters for the first level of clustering. Default is 3.",
    )
    parser.add_argument(
        "--n_clusters_level2",
        type=int,
        default=3,
        help="Number of clusters for the second level of clustering (within each level 1 cluster). Default is 3.",
    )
    parser.add_argument(
        "--method1",
        type=str,
        default="spectral",
        help="Method to use for the first level of clustering. Default is spectral.",
        choices=["spectral", "greedy_comm", "louvain"],
    )
    parser.add_argument(
        "--method2",
        type=str,
        default="spectral",
        help="Method to use for the second level of clustering. Default is spectral.",
        choices=["spectral", "greedy_comm", "louvain"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="aggregated",
        choices=["non_aggregated", "aggregated"],
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
        args.normalize,
        real_data,
        feature_names,
        args.n_clusters_level1,
        args.n_clusters_level2,
        args.mode,
        args.method1,
        args.method2,
        args.distance_metric,
    )
