import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Define the hyperparameters and their possible values
graph_construction_methods = ["flatten"]
distances = ["euclidean", "manhattan", "cosine", "dtw"]
connectivities = ["epsilon", "knn"]
k_neighbors = [3, 5, 7]
percentiles = [10, 20, 30]
clustering_algorithms = ["louvain"]
girvan_num_communities = [None, 2, 3, 4, 5]
girvan_target_modularity = [None, 0.25, 0.3, 0.35, 0.4]
spectral_num_clusters = list(range(2, 10))

# Define model names
model_name = [f"TimeVAE_model{i}" for i in range(7, 69)]

# Generate all combinations of hyperparameters
combinations = list(
    itertools.product(
        graph_construction_methods,
        distances,
        connectivities,
        k_neighbors,
        percentiles,
        clustering_algorithms,
        girvan_num_communities,
        girvan_target_modularity,
        spectral_num_clusters,
        model_name,
    )
)

# To track unique experiments
louvains = []
spectrals = []


# Function to run the script with specific hyperparameters
def run_script(combination):
    (
        graph_construction_method,
        distance,
        connectivity,
        k,
        percentile,
        clustering_algorithm,
        girvan_num_communities,
        girvan_target_modularity,
        spectral_num_clusters,
        model_name,
    ) = combination

    if clustering_algorithm == "louvain":
        if connectivity == "knn":
            if (
                graph_construction_method,
                distance,
                connectivity,
                k,
                model_name,
            ) in louvains:
                return None
            else:
                louvains.append(
                    (graph_construction_method, distance, connectivity, k, model_name)
                )
        elif connectivity == "epsilon":
            if (
                graph_construction_method,
                distance,
                connectivity,
                model_name,
            ) in louvains:
                return None
            else:
                louvains.append(
                    (graph_construction_method, distance, connectivity, model_name)
                )
    elif clustering_algorithm == "spectral":
        if connectivity == "knn":
            if (
                graph_construction_method,
                distance,
                connectivity,
                k,
                spectral_num_clusters,
                model_name,
            ) in spectrals:
                return None
            else:
                spectrals.append(
                    (
                        graph_construction_method,
                        distance,
                        connectivity,
                        k,
                        spectral_num_clusters,
                        model_name,
                    )
                )
        elif connectivity == "epsilon":
            if (
                graph_construction_method,
                distance,
                connectivity,
                spectral_num_clusters,
                model_name,
            ) in spectrals:
                return None
            else:
                spectrals.append(
                    (
                        graph_construction_method,
                        distance,
                        connectivity,
                        spectral_num_clusters,
                        model_name,
                    )
                )
    # print(f"Running for model: {model_name}")
    try:
        cmd = [
            "python",
            "embedded_complex_network.py",
            "--graph_construction_method",
            graph_construction_method,
            "--distance",
            distance,
            "--connectivity",
            connectivity,
            "--k",
            str(k),
            "--percentile",
            str(percentile),
            "--clustering_algorithm",
            clustering_algorithm,
            "--girvan_num_communities",
            str(girvan_num_communities),
            "--girvan_target_modularity",
            str(girvan_target_modularity),
            "--spectral_num_clusters",
            str(spectral_num_clusters),
            "--model_name",
            model_name,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to run with hyperparameters: {combination}")
        return None


# Use ThreadPoolExecutor to run experiments in parallel
print("Total number of combinations:", len(combinations))
results = []
with ThreadPoolExecutor(
    max_workers=8
) as executor:  # Adjust max_workers based on your CPU
    futures = [executor.submit(run_script, combination) for combination in combinations]
    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Processing combinations"
    ):
        result = future.result()
        if result is not None:
            results.append(result)
            print(f"Completed experiment.")
