import torch

import torch.nn as nn
import argparse
from util import load_data
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate marginal distribution comparison"
    )
    parser.add_argument(
        "--model_name", type=str, default="TimeGAN_model3", help="Name of the model"
    )
    model = parser.parse_args().model_name

    # load generated Data
    # Load npy data from the parent folder
    generated_data = np.load(f"../Generated/{model}/generated_samples.npy")

    # Convert the data to torch tensor
    generated_data = torch.from_numpy(generated_data)

    if generated_data.shape[2] == 29:
        if generated_data.shape[1] > 100:
            real_data = load_data(
                "../Data/PreProcessed/29var/df29.xlsx",
                leave_out_problematic_features=False,
                break_to_smaller=False,
            )
        else:
            real_data = load_data(
                "../Data/PreProcessed/29var/df29.xlsx",
                leave_out_problematic_features=False,
                break_to_smaller=True,
                break_size=generated_data.shape[1],
            )

    elif generated_data.shape[2] == 25:
        if generated_data.shape[1] > 100:
            real_data = load_data(
                "../Data/PreProcessed/25var/df25.xlsx",
                leave_out_problematic_features=True,
                break_to_smaller=False,
            )
        else:
            real_data = load_data(
                "../Data/PreProcessed/25var/df25.xlsx",
                leave_out_problematic_features=True,
                break_to_smaller=True,
                break_size=generated_data.shape[1],
            )

    print(generated_data.shape)
    print(real_data.shape)
    # discriminator = Discriminator(input_size, hidden_size, num_layers)
    # train_data = ...  # Replace with your training data
    # labels = ...  # Replace with your target labels
    # num_epochs = ...
    # learning_rate = ...

    # criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # for epoch in range(num_epochs):
    #     outputs = discriminator.forward(train_data)
    #     loss = criterion(outputs, labels)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     if (epoch + 1) % 100 == 0:
    #         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
