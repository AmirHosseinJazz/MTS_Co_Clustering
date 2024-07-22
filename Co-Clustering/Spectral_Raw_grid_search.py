import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


aggregation = ["flatten","mean"]
distance_metric = ["euclidean", "cosine", "manhattan"]
n_clusters = [3, 4, 5, 6, 7, 8, 9, 10]

# Create a list of all possible combinations of hyperparameters
combinations_spectral = list(
    itertools.product(
        distance_metric,
        n_clusters,
        aggregation
    )
)


def run_script_spectral_raw(combination):
    (
        distance_metric,
        n_clusters,
        aggregation
    ) = combination
    try:
        print(f"Running an experiment with {combination}")
        cmd = [
            "python",
            "Spectral_RawData.py",
            "--distance_metric",
            distance_metric,
            "--num_clusters",
            str(n_clusters),
            "--aggregation",
            aggregation,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to run with hyperparameters: {combination}")
        return None



print(
    "Total number of combinations:",
    len(combinations_spectral),
)
results = []
with ThreadPoolExecutor(
    max_workers=8
) as executor:  # Adjust max_workers based on your CPU
    # Create a map of future tasks, and wrap them with tqdm for progress display
    futures = [
        executor.submit(run_script_spectral_raw, combination)
        for combination in combinations_spectral
    ]
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())
