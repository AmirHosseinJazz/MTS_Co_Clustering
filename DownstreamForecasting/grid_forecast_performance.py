import os
import pandas as pd
import itertools
import subprocess

if __name__ == "__main__":
    all_experiments = os.listdir("../Clustering/experiment_embedded_data")
    #  run downstream forecast for all experiments as args
    for _experiment in all_experiments:
        try:
            print(f"Running an experiment with {_experiment}")
            cmd = [
                "python",
                "Downstream_Forecast.py",
                "--experiment_technique",
                "Clustering",
                "--experiment_type",
                "embedded_data",
                "--experiment_name",
                _experiment,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            break
        except Exception as e:
            print(f"Error: {e}")
            print(f"Failed to run with hyperparameters: {_experiment}")
    # all_experiments = os.listdir("../Clustering/experiment_raw_data")
    # #  run downstream forecast for all experiments as args
    # for _experiment in all_experiments:
    #     try:
    #         print(f"Running an experiment with {_experiment}")
    #         cmd = [
    #             "python",
    #             "Downstream_Forecast.py",
    #             "--experiment_technique",
    #             "Clustering",
    #             "--experiment_type",
    #             "raw_data",
    #             "--experiment_name",
    #             _experiment,
    #         ]
    #         result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         print(f"Failed to run with hyperparameters: {_experiment}")

    # all_experiments = os.listdir("../Clustering/experiment_simple_clustering")
    # #  run downstream forecast for all experiments as args
    # for _experiment in all_experiments:
    #     try:
    #         print(f"Running an experiment with {_experiment}")
    #         cmd = [
    #             "python",
    #             "Downstream_Forecast.py",
    #             "--experiment_technique",
    #             "Clustering",
    #             "--experiment_type",
    #             "simple_clustering",
    #             "--experiment_name",
    #             _experiment,
    #         ]
    #         result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         print(f"Failed to run with hyperparameters: {_experiment}")

    # all_experiments = os.listdir("../Co-Clustering/experiment_BiPartite_Embedded")
    # #  run downstream forecast for all experiments as args
    # for _experiment in all_experiments:
    #     try:
    #         print(f"Running an experiment with {_experiment}")
    #         cmd = [
    #             "python",
    #             "Downstream_Forecast.py",
    #             "--experiment_technique",
    #             "Co-Clustering",
    #             "--experiment_type",
    #             "BiPartite_Embedded",
    #             "--experiment_name",
    #             _experiment,
    #         ]
    #         result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         print(f"Failed to run with hyperparameters: {_experiment}")
