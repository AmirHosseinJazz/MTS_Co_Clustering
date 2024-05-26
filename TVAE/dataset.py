import torch
from torch.utils.data import Dataset, DataLoader


class TimeVAEdataset(Dataset):
    def __init__(self, data):
        """
        Assumes data is a numpy array of shape (num_patients, num_timestamps, num_features).
        Converts data to a tensor for PyTorch processing.
        """
        self.data = torch.tensor(data, dtype=torch.float32)

    def __getitem__(self, index):
        return self.data[
            index
        ]  # Returns a tensor of shape (num_timestamps, num_features)

    def __len__(self):
        return len(self.data)
