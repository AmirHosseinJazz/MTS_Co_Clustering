import os
import pandas as pd


def eval():
    # search all folders that include experiment_raw_data
    folders = [
        x[0] for x in os.walk("./experiment_raw_data") if "experiment_raw_data" in x[0]
    ]
    folders_2 = [
        x[0]
        for x in os.walk("./experiment_embedded_data")
        if "experiment_embedded_data" in x[0]
    ]
    folders_3 = [
        x[0]
        for x in os.walk("./experiment_simple_clustering")
        if "experiment_simple_clustering" in x[0]
    ]
    # read all data
    data = []
    for folder in folders:
        try:
            df = pd.read_csv(os.path.join(folder, "info.csv"))
            df = df.T
            df.columns = df.iloc[0]
            df = df[1:]
            df["folder"] = folder
            data.append(df)
            df["Type"]="Graph+Raw"
        except:
            print("error in reading data from", folder)
    for folder in folders_2:
        try:
            df = pd.read_csv(os.path.join(folder, "info.csv"))
            df = df.T
            df.columns = df.iloc[0]
            df = df[1:]
            df["folder"] = folder
            df["Type"] = "Graph+Embedded"
            data.append(df)
        except:
            print("error in reading data from", folder)
    for folder in folders_3:
        try:
            df = pd.read_csv(os.path.join(folder, "info.csv"))
            df = df.T
            df.columns = df.iloc[0]
            df = df[1:]
            df["folder"] = folder
            df["Type"]="Embedded+Clustering"
            data.append(df)
        except:
            print("error in reading data from", folder)
    data = pd.concat(data)
    cols = list(data)
    cols.insert(0, cols.pop(cols.index("Clustering Algorithm")))
    cols.insert(0, cols.pop(cols.index("Type")))
    data = data.loc[:, cols]
    # for the rows with value of Connectivy=epsilon set the k_neighbors to 'N/A'
    data.loc[data["Connectivity"] == "epsilon", "K Neighbors"] = "N/A"
    # for the rows with the value of Clustering Algorithm in [louvain, spectral] set percentiles to 'N/A'
    data.loc[
        data["Clustering Algorithm"].isin(["louvain", "spectral"]), "Percentile"
    ] = "N/A"
    # for the rows with the value of clustering algorithm not equal to girvan_newman set the garvin_num_communities to 'N/A' and girvan_target_modularity to 'N/A'
    data.loc[
        ~data["Clustering Algorithm"].isin(["girvan_newman"]), "girvan_modularity"
    ] = "N/A"
    data.loc[
        ~data["Clustering Algorithm"].isin(["girvan_newman"]), "girvan_communities"
    ] = "N/A"
    data.to_csv("clustering_evaluation.csv", index=False)


if __name__ == "__main__":
    eval()
