import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import silhouette_score, pairwise_distances
import os
from datetime import datetime
import argparse
from collections import Counter
import math
from util import load_data
import ast

def create_dir():
    exp_id = (
        str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        .replace(" ", "")
        .replace(":", "")
    )
    save_dir = f"./experiment_Spectral_Raw/experiment_{exp_id}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def calculate_entropy(labels):
    total_elements = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / total_elements
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_coverage(data, row_labels, column_labels):
    num_row_clusters = len(np.unique(row_labels))
    num_column_clusters = len(np.unique(column_labels))
    
    biclusters = np.zeros((num_row_clusters, num_column_clusters), dtype=bool)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            biclusters[row_labels[i], column_labels[j]] = True
            
    total_elements = data.size
    covered_elements = np.sum(biclusters)
    
    coverage = covered_elements / total_elements
    return coverage

def main(data, distance_metric, n_clusters, aggregation, model_name,feature_names):
    print("Original Data Shape:", data.shape)

    # Apply the specified aggregation method
    if aggregation == "flatten":
        data_processed = data.reshape(data.shape[0], -1)
    elif aggregation == "mean":
        data_processed = np.mean(data, axis=1)
    else:
        raise ValueError("Unsupported aggregation method")

    print("Processed Data Shape:", data_processed.shape)


    # Implement Spectral Biclustering
    model = SpectralBiclustering(n_clusters=n_clusters, method='log', random_state=0)
    model.fit(data_processed)
    row_labels = model.row_labels_
    column_labels = model.column_labels_

    # Save clusters
    save_dir = create_dir()
    with open(f"{save_dir}/results.txt", "w") as f:
        for cluster_idx, cluster in enumerate(np.unique(row_labels)):
            f.write(f"Individual Cluster {cluster_idx}: {np.where(row_labels == cluster)[0].tolist()}\n")
        for cluster_idx, cluster in enumerate(np.unique(column_labels)):
            f.write(f"Feature Cluster {cluster_idx}: {np.where(column_labels == cluster)[0].tolist()}\n")

    with open(f"{save_dir}/clusters.txt", "w") as f:
        for cluster_idx, cluster in enumerate(np.unique(row_labels)):
            f.write(f"Cluster {cluster_idx + 1}: {np.where(row_labels == cluster)[0].tolist()}\n")
        for cluster_idx, cluster in enumerate(np.unique(column_labels)):
            f.write(f"Cluster {cluster_idx + 1}: {np.where(column_labels == cluster)[0].tolist()}\n")
    with open(f"{save_dir}/assignment.txt", "w") as f:
        for sample_idx, cluster in enumerate(row_labels):
            f.write(f"sample {sample_idx + 1} cluster {cluster + 1}\n")

    # Calculate metrics
    distance_matrix_samples = pairwise_distances(data_processed, metric=distance_metric)
    distance_matrix_features = pairwise_distances(data_processed.T, metric=distance_metric)
    coverage = calculate_coverage(data_processed, row_labels, column_labels)
    entropy_samples = calculate_entropy(row_labels)
    entropy_features = calculate_entropy(column_labels)
    silhouette_samples_val = silhouette_score(distance_matrix_samples, row_labels, metric='precomputed')
    silhouette_features_val = silhouette_score(distance_matrix_features, column_labels, metric='precomputed')
    
    metrics_dict = {
        "Method": "SpectralBiclustering",
        "Distance Metric": distance_metric,
        "Number of Clusters": n_clusters,
        "Coverage": coverage,
        "Entropy Samples": entropy_samples,
        "Entropy Features": entropy_features,
        "Silhouette Samples": silhouette_samples_val,
        "Silhouette Features": silhouette_features_val,
        "Model Name": model_name,
    }
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index", columns=["Value"])
    metrics_df.to_csv(f"{save_dir}/metrics.csv")
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
        "--aggregation",
        type=str,
        default="flatten",
        choices=["flatten", "mean"],
        help="Aggregation method to use for clustering. Default is flatten"
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
    main( real_data, args.distance_metric,args.num_clusters,args.aggregation, args.model_name,feature_names)
