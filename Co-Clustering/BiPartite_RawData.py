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
from collections import Counter
import math


def create_dir():
    exp_id = (
        str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        .replace(" ", "")
        .replace(":", "")
    )
    save_dir = f"./experiment_BiPartite/experiment_{exp_id}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def evaluate_modularity(G, partition):
    """Evaluate and print the modularity of the partition on graph G."""
    modularity = community_louvain.modularity(partition, G)
    print(f"Modularity: {modularity}")
    return modularity


def evaluate_silhouette(G, partition):
    """Evaluate and print the silhouette score of the partition. This function assumes that each node's
    community label is its attribute in the graph."""
    if len(set(partition.values())) > 1:
        # Create a simple representation of the graph nodes and their cluster for silhouette score
        nodes, labels = zip(*partition.items())
        # Convert the graph into an edge matrix format for silhouette score calculation
        distances = nx.to_numpy_array(G, nodelist=nodes)
        # Silhouette score calculation
        score = silhouette_score(distances, labels, metric="precomputed")
        print(f"Silhouette Score: {score}")
        return score
    else:
        print("Not enough clusters to calculate Silhouette Score.")
        return None


def calculate_coverage(graph, partition):
    """
    Calculate the coverage of the partition on the given graph.

    Parameters:
    graph (networkx.Graph): The graph.
    partition (dict): A dictionary where keys are nodes and values are the community assigned to each node.

    Returns:
    float: The coverage of the partition.
    """
    intra_community_edges = 0
    for u, v in graph.edges():
        if partition[u] == partition[v]:
            intra_community_edges += 1
    coverage = intra_community_edges / graph.number_of_edges()
    return coverage


def calculate_entropy(partition):
    """
    Calculate the entropy of the partition.

    Parameters:
    partition (dict): A dictionary where keys are nodes and values are the community assigned to each node.

    Returns:
    float: The entropy of the partition.
    """
    total_nodes = len(partition)
    community_counts = Counter(partition.values())
    entropy = 0.0
    for count in community_counts.values():
        p_xi = count / total_nodes
        entropy -= p_xi * math.log2(p_xi)
    return entropy


def community_detection_and_evaluation(G):
    partition = community_louvain.best_partition(G, resolution=1.0)
    modularity = evaluate_modularity(G, partition)
    silhouette = evaluate_silhouette(G, partition)
    coverage = calculate_coverage(G, partition)
    entropy = calculate_entropy(partition)
    results = {
        "graph": G,
        "partition": partition,
        "modularity": modularity,
        "silhouette": silhouette,
    }
    return results


def dtw_distance(ts1, ts2):
    """Compute Dynamic Time Warping distance between two time series"""
    distance, path = fastdtw(ts1, ts2, dist=euclidean)
    return distance


def compute_featurewise_averages(data):
    """
    Compute the average value of each feature for each individual.

    Parameters:
    data (numpy.ndarray): The data array of shape (num_instances, num_timesteps, num_features).

    Returns:
    numpy.ndarray: The average values of shape (num_instances, num_features).
    """
    # Compute the average value of each feature for each individual
    averages = np.mean(data, axis=1)
    return averages


def create_bipartite_graph(data):
    """
    Create a bipartite graph from the given data.

    Parameters:
    data (numpy.ndarray): The data array of shape (num_instances, num_timesteps, num_features).

    Returns:
    networkx.Graph: The bipartite graph.
    """
    num_instances, num_timesteps, num_features = data.shape

    # Compute the average feature values for each individual
    averages = compute_featurewise_averages(data)

    G = nx.Graph()
    G.add_nodes_from(range(num_instances), bipartite=0)  # Individuals
    G.add_nodes_from(
        range(num_instances, num_instances + num_features), bipartite=1
    )  # Features

    # Add edges with weights based on the average feature values
    for i in range(num_instances):
        for j in range(num_features):
            G.add_edge(i, num_instances + j, weight=averages[i, j])

    return G


def main(normalize, data, feature_names, distance_metric):

    # Normalize the data
    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    graphs = create_bipartite_graph(data)
    results = community_detection_and_evaluation(graphs)
    save_dir = create_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(results)

    # for metric, result in results.items():
    #     G = result["graph"]
    #     with open(f"{save_dir}/{metric}_community.txt", "w") as f:
    #         for node, community in result["partition"].items():
    #             f.write(f"{node},{community}\n")
    #     with open(f"{save_dir}/{metric}_modularity.txt", "w") as f:
    #         f.write(f"{result['modularity']}\n")
    #     with open(f"{save_dir}/{metric}_silhouette.txt", "w") as f:
    #         f.write(f"{result['silhouette']}\n")

    #     partition = result["partition"]
    #     plt.figure(figsize=(40, 40))
    #     pos = nx.spring_layout(G)
    #     nx.draw(
    #         G,
    #         pos,
    #         with_labels=True,
    #         node_color=[partition.get(node) for node in G.nodes()],
    #         node_size=50,
    #         cmap="viridis",
    #     )
    #     plt.title(f"Community Graph for {metric}")
    #     plt.savefig(f"{save_dir}/{metric}_community.png")


if __name__ == "__main__":
    # Parse command-line arguments

    parser = argparse.ArgumentParser(
        description="Run community detection on generated data with various configurations."
    )
    parser.add_argument(
        "--normalize",
        type=bool,
        default=False,
        help="Normalize the input data. Default is True.",
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
        choices=["euclidean", "cosine", "manhattan", "dtw"],
        help="Distance metric to use for clustering. Default is euclidean.",
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
    main(args.normalize, real_data, feature_names, args.distance_metric)
