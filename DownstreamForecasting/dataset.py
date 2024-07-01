from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data  # shape: (num_samples, num_timesteps, num_features)
        self.window_size = window_size
        self.X, self.Y = self.create_sliding_windows()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def create_sliding_windows(self):
        X = []
        Y = []

        for sample in self.data:
            for i in range(len(sample) - self.window_size):
                X.append(sample[i : i + self.window_size])
                Y.append(sample[i + self.window_size])

        X = np.array(X)
        Y = np.array(Y)

        return torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float)
