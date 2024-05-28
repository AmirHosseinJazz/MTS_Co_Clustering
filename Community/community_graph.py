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

import community as community_louvain
import argparse
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime


def evaluate_partition(G, partition):
    modularity = community_louvain.modularity(partition, G)
    print(f"Modularity: {modularity}")


def evaluate_partition_sil(data, partition):
    # Converting partition dictionary to a list of labels aligned with data
    labels = [partition.get(i, -1) for i in range(len(data))]

    # Check if there is more than one label and more than zero labels to calculate Silhouette Score
    if len(set(labels)) > 1 and len(set(labels)) < len(data):
        silhouette = silhouette_score(data, labels, metric="euclidean")
        print(f"Silhouette Score: {silhouette}")
    else:
        print(
            "Not enough clusters for Silhouette Score. There must be at least 2 clusters."
        )


def main(distance_metric, connectivity, k_neighbors, epsilon, normalize):
    # Example data: 10 samples, 50 features each
    data = np.random.rand(10, 50)

    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    # Calculate pairwise distances based on the selected metric
    if distance_metric == "euclidean":
        distances = euclidean_distances(data)
    elif distance_metric == "manhattan":
        distances = manhattan_distances(data)
    elif distance_metric == "cosine":
        distances = cosine_distances(data)
    else:
        raise ValueError(
            "Unsupported distance metric. Use 'euclidean', 'manhattan', or 'cosine'."
        )

    # Create a graph
    G = nx.Graph()

    # Edge creation based on the chosen method
    if connectivity == "epsilon":
        threshold = epsilon  # Use user-defined epsilon if epsilon-neighborhood method
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if distances[i][j] < threshold:
                    G.add_edge(i, j, weight=1 / distances[i][j])
    elif connectivity == "knn":
        # Use k-NN graph from sklearn, setting mode to 'distance' to use distances as weights
        knn_graph = kneighbors_graph(
            data,
            n_neighbors=k_neighbors,
            metric=distance_metric,
            mode="distance",
            include_self=False,
        )
        for i, j in zip(*knn_graph.nonzero()):
            G.add_edge(i, j, weight=1 / distances[i][j])

    # Check if the graph has edges
    if G.number_of_edges() == 0:
        print("No edges in the graph. Consider adjusting the connectivity parameters.")
        return

    # Community detection
    partition = community_louvain.best_partition(G)

    # Print the detected communities
    print("Detected communities:", partition)

    evaluate_partition(G, partition)
    evaluate_partition_sil(data, partition)

    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True)
    labels = {i: f"C{partition[i]}" for i in partition}
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    # plt.show()
    ###

    exp_id = (
        str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        .replace(" ", "")
        .replace(":", "")
    )
    if not os.path.exists("./experiment_{}".format(exp_id)):
        os.makedirs("./experiment_{}".format(exp_id))

    plt.savefig("./experiment_{}".format(exp_id) + "/community_graph.png")
    ## save data as csv
    np.savetxt("./experiment_{}/data.csv".format(exp_id), data, delimiter=",")
    ## save partition as csv
    np.savetxt(
        "./experiment_{}/partition.csv".format(exp_id),
        np.array(list(partition.values())),
        delimiter=",",
    )
    ## save graph as gexf
    nx.write_gexf(G, "./experiment_{}/graph.gexf".format(exp_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run community detection on generated data with various configurations."
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan", "cosine"],
        help="The distance metric to use (euclidean, manhattan, cosine). Default is euclidean.",
    )
    parser.add_argument(
        "--connectivity",
        type=str,
        default="knn",
        choices=["epsilon", "knn"],
        help="The method to create graph edges (epsilon for epsilon-neighborhood, knn for k-nearest neighbors). Default is epsilon.",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=5,
        help="Number of neighbors to use for k-NN method. Default is 5.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.5,
        help="Epsilon value for epsilon-neighborhood method. Default is 0.5.",
    )
    parser.add_argument(
        "--normalize",
        type=bool,
        default=True,
        help="Normalize the input data. Default is False.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="TimeGAN_model1",
        help="Name of the model used to generate the data. Default is TimeGAN_model1.",
    )
    args = parser.parse_args()
    main(
        args.distance, args.connectivity, args.k_neighbors, args.epsilon, args.normalize
    )
