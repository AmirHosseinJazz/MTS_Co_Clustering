import torch.nn as nn
import torch
from dataset import SlidingWindowDataset
from torch.utils.data import DataLoader
from util import load_data
import argparse
import pandas as pd
import numpy as np
import os


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # output at the last timestep
        return out


def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device="cpu"):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device).float()  # Convert inputs to float if not already
            targets = targets.to(
                device
            ).float()  # Convert targets to float if not already
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)  # Use targets as ground truth for loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}"
        )


def evaluate_model(model, test_loader, criterion, device="cpu"):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for inputs, targets in test_loader:
            inputs = inputs.to(device).float()  # Convert inputs to float if not already
            targets = targets.to(
                device
            ).float()  # Convert targets to float if not already

            outputs = model(inputs)

            loss = criterion(outputs, targets)  # Calculate loss
            running_loss += loss.item()

        avg_loss = running_loss / len(test_loader)
        print(f"Evaluation Loss: {avg_loss}")
        return avg_loss


def main(experiment_name, experiment_technique, experiment_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load data
    real_data, feature_names = load_data(
        "../Data/PreProcessed/29var/df29.xlsx",
        break_to_smaller=False,
        leave_out_problematic_features=True,
        feature_shape=30,
    )
    print(real_data.shape)
    print(f"Experiment Technique: {experiment_technique}")
    print(f"Experiment Type: {experiment_type}")
    print(f"Experiment Name: {experiment_name}")
    try:
        partitions = pd.read_csv(
            f"../{experiment_technique}/experiment_{experiment_type}/{experiment_name}/assignments.txt",
            sep=" ",
            header=None,
        )
        partitions.columns = ["txt1", "sample", "txt2", "Cluster"]
        partitions.drop(columns=["txt1", "txt2"], inplace=True)
        partitions["sample"] = partitions["sample"].apply(
            lambda x: int(str(x).replace(":", ""))
        )
        partitions["Cluster"] = partitions["Cluster"].apply(
            lambda x: int(str(x).replace(":", ""))
        )
    except Exception as e1:
        print(f"First read attempt failed with error: {e1}")
        try:
            partitions = pd.read_csv(
                f"../{experiment_technique}/experiment_{experiment_type}/{experiment_name}/partition.csv",
                header=None,
                names=["Cluster"],
            )
            partitions["sample"] = partitions.index
            partitions["Cluster"] = partitions["Cluster"].apply(
                lambda x: int(float((str(x).replace(":", ""))))
            )
        except Exception as e2:
            print(f"Second read attempt failed with error: {e2}")
            return
    print(partitions.head())
    print(f"Number of clusters: {partitions['Cluster'].nunique()}")

    clusters = {}
    for i in range(partitions["Cluster"].nunique()):
        clusters[i] = []

    ## if min sample_id is 1, then we need to subtract 1 from sample_id
    if partitions["sample"].min() == 1:
        partitions["sample"] = partitions["sample"] - 1
    for index, row in partitions.iterrows():
        sample_id = row["sample"]
        cluster_id = row["Cluster"]
        clusters[cluster_id].append(real_data[sample_id])

    # print(clusters.keys())
    for i in range(partitions["Cluster"].nunique()):
        clusters[i] = np.array(clusters[i])

    mse_errors = []

    for key, data_cluster in clusters.items():
        print(f"Cluster {key + 1}")
        print(f"Data shape: {data_cluster.shape}")
        if data_cluster.shape[0] <= 5:
            print("Skipping cluster due to insufficient data")
            continue
        dataset = SlidingWindowDataset(data_cluster, window_size=50)
        print(f"Dataset length: {len(dataset)}")

        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        model = LSTMModel(
            input_size=data_cluster.shape[-1],
            hidden_size=32,
            num_layers=1,
            output_size=data_cluster.shape[-1],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        print(f"Training model for cluster {key + 1}")
        train_model(
            model, train_loader, nn.MSELoss(), optimizer, num_epochs=50, device=device
        )

        print(f"Evaluating model for cluster {key + 1}")
        mse_error = evaluate_model(model, test_loader, nn.MSELoss(), device=device)
        mse_errors.append(mse_error)

    if not os.path.exists(f"./evaluation_results/"):
        os.makedirs(f"./evaluation_results/")

    result = pd.DataFrame(mse_errors)
    result.columns = ["MSE"]
    result.to_csv(
        f"./evaluation_results/{experiment_technique}_{experiment_type}_{experiment_name}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run community detection on generated data with various configurations."
    )
    parser.add_argument(
        "--experiment_technique",
        type=str,
        default="Clustering",
        choices=["Clustering", "Co-Clustering"],
        help="Name of the experiment to run",
    )
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="simple_clustering",
        help="Type of experiment to run",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment_simple_clustering_2024-06-13_10-10-29",
        help="Name of the experiment to run",
    )
    args = parser.parse_args()
    main(args.experiment_name, args.experiment_technique, args.experiment_type)
