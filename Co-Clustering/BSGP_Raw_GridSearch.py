import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


n_clusters = [3, 4, 5, 6, 7, 8, 9, 10]
percentile = [60, 65, 70, 75, 80, 85, 90, 95]
distance = ["euclidean", "cosine", "manhattan"]


# Create a list of all possible combinations of hyperparameters
combinations_BSGP_raw = list(
    itertools.product(
        n_clusters,
        percentile,
        distance,
    )
)


def run_script_BSGP_raw(combination):
    (n_clusters, percentile, distance) = combination
    try:
        print(f"Running an experiment with {combination}")
        cmd = [
            "python",
            "BSGP_Raw.py",
            "--num_clusters",
            str(n_clusters),
            "--percentile",
            str(percentile),
            "--distance_metric",
            distance,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to run with hyperparameters: {combination}")
        return None


print(
    "Total number of combinations:",
    len(combinations_BSGP_raw),
)
results = []
with ThreadPoolExecutor(
    max_workers=8
) as executor:  # Adjust max_workers based on your CPU
    futures = [
        executor.submit(run_script_BSGP_raw, combination)
        for combination in combinations_BSGP_raw
    ]
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())
