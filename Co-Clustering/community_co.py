import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import community
import os
from datetime import datetime
import argparse
from util import load_data
import ast
from scipy.stats import skew, kurtosis


def main(latent_vectors, aggregated_features, aggregated_features_names):
    # Create a bipartite graph
    B = nx.Graph()

    # take first 10 samples from latent vectors
    # latent_vectors = latent_vectors[:10]
    # aggregated_features = aggregated_features[:10]
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

    vertical_spacing = 10  # change the spacing value as needed for better visibility
    pos = {}

    for index, individual in enumerate(individuals):
        pos[individual] = (-100, index * vertical_spacing)  # Column 0 for individuals

    for index, feature in enumerate(features):
        pos[feature] = (100, index * vertical_spacing)  # Column 1 for features

    node_colors = ["blue" if B.nodes[node]["bipartite"] == 0 else "green" for node in B]

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

    exp_id = (
        str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        .replace(" ", "")
        .replace(":", "")
    )
    save_dir = f"./experiment_{exp_id}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

    # with open(f"{save_dir}/community_nodes.txt", "w") as f:
    #     f.write(f"Community Nodes: {community_nodes}\n")

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
        default="TimeGAN_model1",
        help="Name of the model used to generate the data. Default is TimeGAN_model1.",
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
    main(latent_vectors, aggregated_features, aggregated_features_names[:])
