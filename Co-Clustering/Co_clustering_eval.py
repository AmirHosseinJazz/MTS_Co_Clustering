import pandas as pd
import numpy as np
import os


if __name__ == "__main__":
    all_experiments = []
    expirements_bi_emb = []
    experiments_bi_raw = []
    try:
        for k in os.listdir("./experiment_BiPartite_Embedded"):
            for t in os.listdir("./experiment_BiPartite_Embedded/" + k):
                if "metrics.csv" in t:
                    df = pd.read_csv("./experiment_BiPartite_Embedded/" + k + "/" + t)
                    df = df.T
                    df.columns = df.iloc[0]
                    df = df.iloc[1:]
                    expirements_bi_emb.append(df)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to read the file")
    try:
        for k in os.listdir("./experiment_BiPartite"):
            for t in os.listdir("./experiment_BiPartite/" + k):
                if "metrics.csv" in t:
                    df = pd.read_csv("./experiment_BiPartite/" + k + "/" + t)
                    df = df.T
                    df.columns = df.iloc[0]
                    df = df.iloc[1:]
                    experiments_bi_raw.append(df)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to read the file")

    try:
        expirements_bi_emb = pd.concat(expirements_bi_emb)
        expirements_bi_emb["type"] = "BiPartite_Embedded"
        all_experiments.append(expirements_bi_emb)
    except:
        pass
    try:
        experiments_bi_raw = pd.concat(experiments_bi_raw)
        experiments_bi_raw["type"] = "BiPartite_Raw"
        all_experiments.append(experiments_bi_raw)
    except:
        pass

    pd.concat(all_experiments).to_csv("co_clustering_evaluation.csv")
    print("Done")
