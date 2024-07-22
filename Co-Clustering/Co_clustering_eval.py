import pandas as pd
import numpy as np
import os


if __name__ == "__main__":
    all_experiments = []
    expirements_spec_emb = []
    experiments_spec_raw = []
    expirements_BSGP_emb = []
    experiments_BSGP_raw = []
    try:
        for k in os.listdir("./experiment_BSGP_Embedded"):
            for t in os.listdir("./experiment_BSGP_Embedded/" + k):
                if "metrics.csv" in t:
                    df = pd.read_csv("./experiment_BSGP_Embedded/" + k + "/" + t)
                    df = df.T
                    df.columns = df.iloc[0]
                    df = df.iloc[1:]
                    expirements_BSGP_emb.append(df)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to read the file")

    try:
        for k in os.listdir("./experiment_Spectral_Embedded"):
            for t in os.listdir("./experiment_Spectral_Embedded/" + k):
                if "metrics.csv" in t:
                    df = pd.read_csv("./experiment_Spectral_Embedded/" + k + "/" + t)
                    df = df.T
                    df.columns = df.iloc[0]
                    df.rename(columns={"Model_name": "Model Name"}, inplace=True)
                    df = df.iloc[1:]
                    expirements_spec_emb.append(df)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to read the file")
        
    
    try:
        for k in os.listdir("./experiment_BSGP_Raw"):
            for t in os.listdir("./experiment_BSGP_Raw/" + k):
                if "metrics.csv" in t:
                    df = pd.read_csv("./experiment_BSGP_Raw/" + k + "/" + t)
                    df = df.T
                    df.columns = df.iloc[0]
                    df = df.iloc[1:]
                    experiments_BSGP_raw.append(df)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to read the file")
    
    try:
        for k in os.listdir("./experiment_Spectral_Raw"):
            for t in os.listdir("./experiment_Spectral_Raw/" + k):
                if "metrics.csv" in t:
                    df = pd.read_csv("./experiment_Spectral_Raw/" + k + "/" + t)
                    df = df.T
                    df.columns = df.iloc[0]
                    df = df.iloc[1:]
                    experiments_spec_raw.append(df)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to read the file")


    try:
        expirements_BSGP_emb = pd.concat(expirements_BSGP_emb)
        expirements_BSGP_emb["type"] = "BSGP_Embedded"
        all_experiments.append(expirements_BSGP_emb)
    except:
        pass

    try:
        expirements_spec_emb = pd.concat(expirements_spec_emb)
        expirements_spec_emb["type"] = "Spectral_Embedded"
        all_experiments.append(expirements_spec_emb)
    except:
        pass

    try:
        experiments_BSGP_raw = pd.concat(experiments_BSGP_raw)
        experiments_BSGP_raw["type"] = "BSGP_Raw"
        all_experiments.append(experiments_BSGP_raw)
    except:
        pass

    
    try:
        experiments_spec_raw = pd.concat(experiments_spec_raw)
        experiments_spec_raw["type"] = "Spectral_Raw"
        all_experiments.append(experiments_spec_raw)
    except:
        pass


    pd.concat(all_experiments).to_csv("co_clustering_evaluation.csv")
    print("Done")
