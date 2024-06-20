import numpy as np
import os
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
import pandas as pd
import argparse


## Main Function
def main(
    clustering_algorithm: str,
    k_means_num_clusters: int,
    k_means_num_init: int,
    hierarchical_num_clusters: int,
    hierarchical_linkage: str,
    dbscan_eps: float,
    dbscan_min_samples: int,
    spectral_num_clusters: int,
    spectral_affinity: str,
    model_name: str,
):
    # Load encoded data
    data = np.load(f"../Generated/{model_name}/encoded_data.npy")

    print(f"Data shape: {data.shape}")
    print(f"Latent space shape: {data.shape[1]}")

    print("Loading Experiment for model: ", model_name)

    # Store the experiment data
    exp_id = (
        str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        .replace(" ", "")
        .replace(":", "")
    )
    save_dir = f"./experiment_simple_clustering/experiment_simple_clustering_{exp_id}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    info_pd = pd.DataFrame(columns=["Item", "Value"])
    info_pd.loc[0] = ["Graph Construction Method", "No_Graph"]
    info_pd.loc[1] = ["Distance Metric", "N/A"]
    info_pd.loc[2] = ["Connectivity", "N/A"]
    info_pd.loc[3] = ["K Neighbors", "N/A"]
    info_pd.loc[4] = ["Percentile", "N/A"]
    info_pd.loc[5] = ["Clustering Algorithm", clustering_algorithm]
    info_pd.loc[6] = ["Model Name", model_name]

    if clustering_algorithm == "k_means":
        print(f"Running Kmeans Algorithm...")
        kmeans = KMeans(n_clusters=k_means_num_clusters, n_init=k_means_num_init)
        kmeans_clusters = kmeans.fit_predict(data)
        kmeans_assignments = [
            (idx, cluster) for idx, cluster in enumerate(kmeans_clusters)
        ]

        with open(f"{save_dir}/assignments.txt", "w") as f:
            for item in kmeans_assignments:
                f.write(f"sample {item[0]}: cluster {item[1]}")
                f.write("\n")

        info_pd.loc[7] = ["Modularity", "N/A"]
        info_pd.loc[8] = [
            "Silhouette Score",
            silhouette_score(data, kmeans_clusters),
        ]
        info_pd.loc[9] = ["Davies Bouldin Index", "N/A"]
        info_pd.loc[10] = ["Num Communities", "N/A"]
        info_pd.loc[11] = ["K Means Num Clusters", k_means_num_clusters]
        info_pd.loc[12] = ["K Means Num Init", k_means_num_init]
        info_pd.to_csv(f"{save_dir}/info.csv", index=False)
    elif clustering_algorithm == "hierarchical":
        print(f"Running Hierarchical Algorithm...")
        hiearchical = AgglomerativeClustering(
            n_clusters=hierarchical_num_clusters, linkage=hierarchical_linkage
        )
        hiearchical_clusters = hiearchical.fit_predict(data)
        hiearchical_assignments = [
            (idx, cluster) for idx, cluster in enumerate(hiearchical_clusters)
        ]

        with open(f"{save_dir}/assignments.txt", "w") as f:
            for item in hiearchical_assignments:
                f.write(f"sample {item[0]}: cluster {item[1]}")
                f.write("\n")
        info_pd.loc[7] = ["Modularity", "N/A"]
        info_pd.loc[8] = [
            "Silhouette Score",
            silhouette_score(data, hiearchical_clusters),
        ]
        info_pd.loc[9] = ["Davies Bouldin Index", "N/A"]
        info_pd.loc[10] = ["Num Communities", "N/A"]
        info_pd.loc[11] = ["Hierarchical Num Clusters", hierarchical_num_clusters]
        info_pd.loc[12] = ["Hierarchical Linkage", hierarchical_linkage]
        info_pd.to_csv(f"{save_dir}/info.csv", index=False)
    elif clustering_algorithm == "dbscan":
        print(f"Running DBSCAN Algorithm...")
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        dbscan_clusters = dbscan.fit_predict(data)
        dbscan_assignments = [
            (idx, cluster) for idx, cluster in enumerate(dbscan_clusters)
        ]

        with open(f"{save_dir}/assignments.txt", "w") as f:
            for item in dbscan_assignments:
                f.write(f"sample {item[0]}: cluster {item[1]}")
                f.write("\n")
        info_pd.loc[7] = ["Modularity", "N/A"]
        if np.any(dbscan_clusters == -1):
            core_samples_mask = np.zeros_like(dbscan_clusters, dtype=bool)
            core_samples_mask[dbscan.core_sample_indices_] = True
            dbscan_data = data[core_samples_mask]
            dbscan_silhouette = (
                silhouette_score(dbscan_data, dbscan_clusters[core_samples_mask])
                if len(set(dbscan_clusters[core_samples_mask])) > 1
                else "Undefined"
            )
        else:
            dbscan_silhouette = (
                silhouette_score(data, dbscan_clusters)
                if len(set(dbscan_clusters)) > 1
                else "Undefined"
            )
        info_pd.loc[8] = ["Silhouette Score", dbscan_silhouette]
        info_pd.loc[9] = ["Davies Bouldin Index", "N/A"]
        info_pd.loc[10] = ["Num Communities", "N/A"]
        info_pd.loc[11] = ["DBSCAN Eps", dbscan_eps]
        info_pd.loc[12] = ["DBSCAN Min Samples", dbscan_min_samples]
        info_pd.to_csv(f"{save_dir}/info.csv", index=False)
    elif clustering_algorithm == "spectral":
        print(f"Running Spectral Algorithm...")
        spectral = SpectralClustering(
            n_clusters=spectral_num_clusters, affinity=spectral_affinity
        )
        spectral_clusters = spectral.fit_predict(data)
        spectral_assignments = [
            (idx, cluster) for idx, cluster in enumerate(spectral_clusters)
        ]

        with open(f"{save_dir}/assignments.txt", "w") as f:
            for item in spectral_assignments:
                f.write(f"sample {item[0]}: cluster {item[1]}")
                f.write("\n")
        info_pd.loc[7] = ["Modularity", "N/A"]
        info_pd.loc[8] = [
            "Silhouette Score",
            silhouette_score(data, spectral_clusters),
        ]
        info_pd.loc[9] = ["Davies Bouldin Index", "N/A"]
        info_pd.loc[10] = ["Num Communities", "N/A"]
        info_pd.loc[11] = ["Spectral Num Clusters", spectral_num_clusters]
        info_pd.loc[12] = ["Spectral Affinity", spectral_affinity]
        info_pd.to_csv(f"{save_dir}/info.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run clustering on embedded data with various configurations."
    )
    parser.add_argument(
        "--clustering_algorithm",
        type=str,
        default="k_means",
        help="Clustering algorithm to use. Default is k_means.",
    )
    parser.add_argument(
        "--k_means_num_clusters",
        type=int,
        default=5,
        help="Number of clusters for KMeans. Default is 5.",
    )
    parser.add_argument(
        "--k_means_num_init",
        type=int,
        default=10,
        help="Number of initializations for KMeans. Default is 10.",
    )
    parser.add_argument(
        "--hierarchical_num_clusters",
        type=int,
        default=5,
        help="Number of clusters for Hierarchical. Default is 5.",
    )
    parser.add_argument(
        "--hierarchical_linkage",
        type=str,
        default="ward",
        help="Linkage method for Hierarchical. Default is ward.",
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=0.5,
        help="Epsilon value for DBSCAN. Default is 0.5.",
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=5,
        help="Minimum samples for DBSCAN. Default is 5.",
    )
    parser.add_argument(
        "--spectral_num_clusters",
        type=int,
        default=5,
        help="Number of clusters for Spectral. Default is 5.",
    )
    parser.add_argument(
        "--spectral_affinity",
        type=str,
        default="nearest_neighbors",
        help="Affinity method for Spectral. Default is nearest_neighbors.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="TimeVAE_model8",
        help="Name of the model used to generate the data. Default is TimeVAE_model8.",
    )
    args = parser.parse_args()

    main(
        args.clustering_algorithm,
        args.k_means_num_clusters,
        args.k_means_num_init,
        args.hierarchical_num_clusters,
        args.hierarchical_linkage,
        args.dbscan_eps,
        args.dbscan_min_samples,
        args.spectral_num_clusters,
        args.spectral_affinity,
        args.model_name,
    )
