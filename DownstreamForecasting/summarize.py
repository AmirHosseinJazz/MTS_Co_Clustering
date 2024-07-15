import pandas as pd
import os
import re


def summarize_results():
    results = []
    for k in os.listdir("./evaluation_results"):
        if k.split("_")[0] == "Clustering":
            if "simple_clustering" in k:
                temp = pd.read_csv(f"./evaluation_results/{k}")
                # take date out of k regex
                pattern = r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})"
                match = re.search(pattern, k)
                if match:
                    date = match.group(0)
                else:
                    date = "None"
                results.append(
                    {
                        "Technique": "Clustering",
                        "Type": "simple_clustering",
                        "Date": date,
                        "MSE": temp["MSE"].mean(),
                    }
                )
            elif "raw_data" in k:
                temp = pd.read_csv(f"./evaluation_results/{k}")
                # take date out of k regex
                pattern = r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})"
                match = re.search(pattern, k)
                if match:
                    date = match.group(0)
                else:
                    date = "None"
                results.append(
                    {
                        "Technique": "Clustering",
                        "Type": "Complex_Network_Raw_Data",
                        "Date": date,
                        "MSE": temp["MSE"].mean(),
                    }
                )
            elif "embedded" in k:
                temp = pd.read_csv(f"./evaluation_results/{k}")
                # take date out of k regex
                pattern = r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})"
                match = re.search(pattern, k)
                if match:
                    date = match.group(0)
                else:
                    date = "None"
                results.append(
                    {
                        "Technique": "Clustering",
                        "Type": "Complex_Network_Embedded",
                        "Date": date,
                        "MSE": temp["MSE"].mean(),
                    }
                )
        if k.split("_")[0] == "Co-Clustering":
            if "Embedded" in k:
                temp = pd.read_csv(f"./evaluation_results/{k}")
                # take date out of k regex
                pattern = r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})"
                match = re.search(pattern, k)
                if match:
                    date = match.group(0)
                else:
                    date = "None"
                results.append(
                    {
                        "Technique": "Co-Clustering",
                        "Type": "Embedded",
                        "Date": date,
                        "MSE": temp["MSE"].mean(),
                    }
                )

    results = pd.DataFrame(results)
    results.to_csv("./summary.csv", index=False)


summarize_results()
