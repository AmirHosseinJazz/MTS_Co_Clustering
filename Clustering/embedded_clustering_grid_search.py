import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Define the hyperparameters and their possible values
#  parser = argparse.ArgumentParser(
#         description="Run clustering on embedded data with various configurations."
#     )
#     parser.add_argument(
#         "--clustering_algorithm",
#         type=str,
#         default="k_means",
#         help="Clustering algorithm to use. Default is k_means.",
#     )
#     parser.add_argument(
#         "--k_means_num_clusters",
#         type=int,
#         default=5,
#         help="Number of clusters for KMeans. Default is 5.",
#     )
#     parser.add_argument(
#         "--k_means_num_init",
#         type=int,
#         default=10,
#         help="Number of initializations for KMeans. Default is 10.",
#     )
#     parser.add_argument(
#         "--hiearchical_num_clusters",
#         type=int,
#         default=5,
#         help="Number of clusters for Hierarchical. Default is 5.",
#     )
#     parser.add_argument(
#         "--hiearchical_linkage",
#         type=str,
#         default="ward",
#         help="Linkage method for Hierarchical. Default is ward.",
#     )
#     parser.add_argument(
#         "--dbscan_eps",
#         type=float,
#         default=0.5,
#         help="Epsilon value for DBSCAN. Default is 0.5.",
#     )
#     parser.add_argument(
#         "--dbscan_min_samples",
#         type=int,
#         default=5,
#         help="Minimum samples for DBSCAN. Default is 5.",
#     )
#     parser.add_argument(
#         "--spectral_num_clusters",
#         type=int,
#         default=5,
#         help="Number of clusters for Spectral. Default is 5.",
#     )
#     parser.add_argument(
#         "--spectral_affinity",
#         type=str,
#         default="nearest_neighbors",
#         help="Affinity method for Spectral. Default is nearest_neighbors.",
#     )

#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="TimeVAE_model9",
#         help="Name of the model used to generate the data. Default is TimeVAE_model9.",
#     )
#     args = parser.parse_args()

#     main(
#         args.clustering_algorithm,
#         args.k_means_num_clusters,
#         args.k_means_num_init,
#         args.hiearchical_num_clusters,
#         args.hiearchical_linkage,
#         args.dbscan_eps,
#         args.dbscan_min_samples,
#         args.spectral_num_clusters,
#         args.spectral_affinity,
#         args.model_name,
#     )
clustering_algorithms = ["k_means", "hierarchical", "dbscan", "spectral"]
k_means_num_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
k_means_num_init = [20]

hierarchical_num_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
hierarchical_linkage = ["ward", "complete", "average", "single"]

dbscan_eps = [
    5,
    5.5,
    6,
    6.5,
    7,
    7.5,
    8,
    8.5,
    9,
    9.5,
    10,
    10.5,
    11,
    11.5,
    12,
]
dbscan_min_samples = [2, 3, 4, 5, 6, 7, 8, 9, 10]
db_scan_metric = ["euclidean", "manhattan", "cosine"]


spectral_num_clusters = list(range(2, 25))
spectral_affinity = ["nearest_neighbors", "rbf"]

# 1 through 68
model_name = [
    "TimeVAE_model7",
    "TimeVAE_model8",
    "TimeVAE_model9",
    "TimeVAE_model10",
    "TimeVAE_model11",
    "TimeVAE_model12",
    "TimeVAE_model13",
    "TimeVAE_model14",
    "TimeVAE_model15",
    "TimeVAE_model16",
    "TimeVAE_model17",
    "TimeVAE_model18",
    "TimeVAE_model19",
    "TimeVAE_model20",
    "TimeVAE_model21",
    "TimeVAE_model22",
    "TimeVAE_model23",
    "TimeVAE_model24",
    "TimeVAE_model25",
    "TimeVAE_model26",
    "TimeVAE_model27",
    "TimeVAE_model28",
    "TimeVAE_model29",
    "TimeVAE_model30",
    "TimeVAE_model31",
    "TimeVAE_model32",
    "TimeVAE_model33",
    "TimeVAE_model34",
    "TimeVAE_model35",
    "TimeVAE_model36",
    "TimeVAE_model37",
    "TimeVAE_model38",
    "TimeVAE_model39",
    "TimeVAE_model40",
    "TimeVAE_model41",
    "TimeVAE_model42",
    "TimeVAE_model43",
    "TimeVAE_model44",
    "TimeVAE_model45",
    "TimeVAE_model46",
    "TimeVAE_model47",
    "TimeVAE_model48",
    "TimeVAE_model49",
    "TimeVAE_model50",
    "TimeVAE_model51",
    "TimeVAE_model52",
    "TimeVAE_model53",
    "TimeVAE_model54",
    "TimeVAE_model55",
    "TimeVAE_model56",
    "TimeVAE_model57",
    "TimeVAE_model58",
    "TimeVAE_model59",
    "TimeVAE_model60",
    "TimeVAE_model61",
    "TimeVAE_model62",
    "TimeVAE_model63",
    "TimeVAE_model64",
    "TimeVAE_model65",
    "TimeVAE_model66",
    "TimeVAE_model67",
    "TimeVAE_model68",
]


# Generate all combinations of hyperparameters k means
combinations_kmeans = list(
    itertools.product(
        # "k_means",
        k_means_num_clusters,
        k_means_num_init,
        model_name,
    )
)

combinations_hierarchical = list(
    itertools.product(
        # "hierarchical",
        hierarchical_num_clusters,
        hierarchical_linkage,
        model_name,
    )
)

combinations_dbscan = list(
    itertools.product(
        # "dbscan",
        dbscan_eps,
        dbscan_min_samples,
        db_scan_metric,
        model_name,
    )
)

combinations_spectral = list(
    itertools.product(
        # "spectral",
        spectral_num_clusters,
        spectral_affinity,
        model_name,
    )
)


def run_script_kmeans(combination):
    (
        # clustering_algorithm,
        k_means_num_clusters,
        k_means_num_init,
        model_name,
    ) = combination
    try:
        print(f"Running an experiment with {combination}")
        cmd = [
            "python",
            "embedded_clustering.py",
            "--clustering_algorithm",
            "k_means",
            "--k_means_num_clusters",
            str(k_means_num_clusters),
            "--k_means_num_init",
            str(k_means_num_init),
            "--model_name",
            model_name,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to run with hyperparameters: {combination}")
        return None


def run_script_hierarchical(combination):
    (
        # clustering_algorithm,
        hierarchical_num_clusters,
        hierarchical_linkage,
        model_name,
    ) = combination
    try:
        print(f"Running an experiment with {combination}")
        cmd = [
            "python",
            "embedded_clustering.py",
            "--clustering_algorithm",
            "hierarchical",
            "--hierarchical_num_clusters",
            str(hierarchical_num_clusters),
            "--hierarchical_linkage",
            hierarchical_linkage,
            "--model_name",
            # "TimeVAE_model9",
            model_name,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to run with hyperparameters: {combination}")
        return None


def run_script_dbscan(combination):
    (
        # clustering_algorithm,
        dbscan_eps,
        dbscan_min_samples,
        db_scan_metric,
        model_name,
    ) = combination
    try:
        print(f"Running an experiment with {combination}")
        cmd = [
            "python",
            "embedded_clustering.py",
            "--clustering_algorithm",
            # clustering_algorithm,
            "dbscan",
            "--dbscan_eps",
            str(dbscan_eps),
            "--dbscan_min_samples",
            str(dbscan_min_samples),
            "--db_scan_metric",
            str(db_scan_metric),
            "--model_name",
            model_name,
            # "TimeVAE_model9",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to run with hyperparameters: {combination}")
        return None


def run_script_spectral(combination):
    (
        # clustering_algorithm,
        spectral_num_clusters,
        spectral_affinity,
        model_name,
    ) = combination
    try:
        print(f"Running an experiment with {combination}")
        cmd = [
            "python",
            "embedded_clustering.py",
            "--clustering_algorithm",
            # clustering_algorithm,
            "spectral",
            "--spectral_num_clusters",
            str(spectral_num_clusters),
            "--spectral_affinity",
            spectral_affinity,
            "--model_name",
            model_name,
            # "TimeVAE_model9",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to run with hyperparameters: {combination}")
        return None


# Use ThreadPoolExecutor to run experiments in parallel
print(
    "Total number of combinations:",
    len(combinations_kmeans)
    + len(combinations_hierarchical)
    + len(combinations_dbscan)
    + len(combinations_spectral),
)
results = []
with ThreadPoolExecutor(
    max_workers=8
) as executor:  # Adjust max_workers based on your CPU
    # Create a map of future tasks, and wrap them with tqdm for progress display
    # futures = [
    #     executor.submit(run_script_kmeans, combination)
    #     for combination in combinations_kmeans
    # ]
    # for future in tqdm(as_completed(futures), total=len(futures)):
    #     results.append(future.result())
    # futures = [
    #     executor.submit(run_script_hierarchical, combination)
    #     for combination in combinations_hierarchical
    # ]
    # for future in tqdm(as_completed(futures), total=len(futures)):
    #     results.append(future.result())
    # futures = [
    #     executor.submit(run_script_dbscan, combination)
    #     for combination in combinations_dbscan
    # ]
    # for future in tqdm(as_completed(futures), total=len(futures)):
    #     results.append(future.result())
    futures = [
        executor.submit(run_script_spectral, combination)
        for combination in combinations_spectral
    ]
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())
