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
k_means_num_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
k_means_num_init = [5, 10, 15, 20]
hierarchical_num_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
hierarchical_linkage = ["ward", "complete", "average", "single"]
dbscan_eps = [0.1, 0.2, 0.3, 0.4, 0.5]
dbscan_min_samples = [2, 3, 4, 5, 6, 7, 8, 9, 10]
spectral_num_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
spectral_affinity = ["nearest_neighbors", "rbf"]
model_name = ["TimeVAE_model9"]



# Generate all combinations of hyperparameters k means
combinations_kmeans = list(
    itertools.product(
        # "k_means",
        k_means_num_clusters,
        k_means_num_init,
        # model_name,
    )
)

combinations_hierarchical = list(
    itertools.product(
        # "hierarchical",
        hierarchical_num_clusters,
        hierarchical_linkage,
        # model_name,
    )
)

combinations_dbscan = list(
    itertools.product(
        # "dbscan",
        dbscan_eps,
        dbscan_min_samples,
        # model_name,
    )
)

combinations_spectral = list(
    itertools.product(
        # "spectral",
        spectral_num_clusters,
        spectral_affinity,
        # model_name,
    )
)
def run_script_kmeans(combination):
    (
        # clustering_algorithm,
        k_means_num_clusters,
        k_means_num_init,
        # model_name,
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
            "TimeVAE_model9",
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
        # model_name,
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
            "TimeVAE_model9",
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
        # model_name,
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
            "--model_name",
            # model_name,
            "TimeVAE_model9",
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
        # model_name,
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
            # model_name,
            "TimeVAE_model9",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to run with hyperparameters: {combination}")
        return None


# Use ThreadPoolExecutor to run experiments in parallel
print("Total number of combinations:", len(combinations_kmeans)+len(combinations_hierarchical)+len(combinations_dbscan)+len(combinations_spectral))
results = []
with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on your CPU
    # Create a map of future tasks, and wrap them with tqdm for progress display
    futures = [executor.submit(run_script_kmeans, combination) for combination in combinations_kmeans]
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())
    futures = [executor.submit(run_script_hierarchical, combination) for combination in combinations_hierarchical]
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())
    futures = [executor.submit(run_script_dbscan, combination) for combination in combinations_dbscan]
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())
    futures = [executor.submit(run_script_spectral, combination) for combination in combinations_spectral]
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())
    
    