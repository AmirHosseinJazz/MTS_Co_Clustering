import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from util import load_data
import numpy as np
import ast
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class SimpleLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(SimpleLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        out = self.sigmoid(out)
        return out


def main(
    model_name="TimeVAE_model1",
    num_epochs=10,
    learning_rate=0.001,
):

    print(f"model_name: {model_name}")
    params = {}
    if "GAN" in model_name:

        try:
            with open(f"../TGAN/saved_models/{model_name}/config.txt", "r") as f:
                for line in f:
                    key, value = line.split(":")
                    # check if the parameter is an integer
                    try:
                        params[key.strip()] = int(value.strip())
                    except:
                        params[key.strip()] = value.strip()
        except Exception as E:
            raise Exception(f"Could not load the configuration file: {E}")
    elif "VAE" in model_name:
        try:
            with open(f"../TVAE/saved_models/{model_name}/config.txt", "r") as f:
                for line in f:
                    key, value = line.split(":")
                    # check if the parameter is an integer
                    try:
                        params[key.strip()] = int(value.strip())
                    except:
                        params[key.strip()] = value.strip()
        except Exception as E:
            raise Exception(f"Could not load the configuration file: {E}")

    # Read model config from a text file
    model_config = {}
    config_file = (
        f"../TGAN/saved_models/{model_name}/config.txt"
        if "GAN" in model_name
        else f"../TVAE/saved_models/{model_name}/config.txt"
    )
    try:
        with open(config_file, "r") as f:
            for line in f:
                key, value = line.split(":")
                # check if the parameter is an integer
                try:
                    model_config[key.strip()] = int(value.strip())
                except:
                    model_config[key.strip()] = value.strip()
    except Exception as E:
        raise Exception(f"Could not load the configuration file: {E}")
    # print(model_config)
    generated_data = np.load(f"../Generated/{model_name}/generated_samples.npy")

    if model_config["data_source"] == "29var":
        real_data, feature_names = load_data(
            "../Data/PreProcessed/29var/df29.xlsx",
            break_to_smaller=ast.literal_eval(model_config["break_data"]),
            break_size=model_config["break_size"],
            leave_out_problematic_features=ast.literal_eval(
                model_config["leave_out_problematic_features"]
            ),
            cutoff_data=model_config["cutoff_data"],
            feature_shape=generated_data.shape[-1],
        )
    elif model_config["data_source"] == "12var":
        real_data, feature_names = load_data(
            "../Data/PreProcessed/12var/df12.xlsx",
            break_to_smaller=ast.literal_eval(model_config["break_data"]),
            break_size=model_config["break_size"],
            leave_out_problematic_features=ast.literal_eval(
                model_config["leave_out_problematic_features"]
            ),
            cutoff_data=model_config["cutoff_data"],
        )

    # Concatenate the real and generated data
    data = np.concatenate((real_data, generated_data), axis=0)

    # Create labels
    labels_real = np.ones((real_data.shape[0],))  # Labels for real data
    labels_generated = np.zeros((generated_data.shape[0],))  # Labels for generated data
    labels = np.concatenate((labels_real, labels_generated), axis=0)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, shuffle=True
    )
    # Convert arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    # Define batch size
    batch_size = 32

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleLSTMClassifier(
        input_dim=real_data.shape[-1], hidden_dim=64, num_layers=1, num_classes=2
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (sequences, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )
    # Evaluate the model

    model.eval()

    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in test_loader:
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not os.path.exists(f"./predictive_measures/{model_name}"):
        os.makedirs(f"./predictive_measures/{model_name}")

    cm = confusion_matrix(all_labels, all_predictions)
    # Calculate precision, recall, and F1 score
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)

    # Save the confusion matrix
    np.savetxt(
        f"./predictive_measures/{model_name}/confusion_matrix.csv",
        cm,
        delimiter=",",
    )

    # store the precision, recall and F1 score
    with open(f"./predictive_measures/{model_name}/f1_recall_precision.txt", "w") as f:
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1_score}\n")

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Generated", "Real"],
        yticklabels=["Generated", "Real"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Labels")
    plt.xlabel("Predicted Labels")
    plt.savefig(f"./predictive_measures/{model_name}/confusion_matrix.png")

    # Save the model

    torch.save(
        model.state_dict(),
        f"./predictive_measures/{model_name}/generated_data_model.pth",
    )
