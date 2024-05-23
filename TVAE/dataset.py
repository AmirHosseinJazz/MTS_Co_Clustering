# -*- coding: UTF-8 -*-
import numpy as np
import torch

class TimeVAEdataset(torch.utils.data.Dataset):
    def __init__(self, data, temporal_data):
        """
        Assumes data is a NumPy array that needs to be converted to a tensor.
        temporal_data is assumed to be a list or a similarly indexable collection.
        """
        self.data = torch.tensor(data, dtype=torch.float32)  # Convert data to float tensors
        self.temporal_data = torch.tensor(temporal_data, dtype=torch.long)  # Convert temporal data to long tensors

    def __getitem__(self, index):
        return self.data[index], self.temporal_data[index]

    def __len__(self):
        return len(self.data)


    def collate_fn(self, batch):
        """Minibatch sampling
        """
        # Pad sequences to max length
        X_mb = [X for X in batch[0]]
        
        # The actual length of each data
        T_mb = [T for T in batch[1]]
        
        return X_mb, T_mb
