import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import wandb
from util import load_data
from TimeVAE import BaseVariationalAutoencoder
import argparse


def main(params):
    if params['data_source']=='29var':
        data=load_data('../Data/PreProcessed/29var/df29.xlsx')
    elif params['data_source']=='12var':
        data=load_data('../Data/PreProcessed/12var/df12.xlsx')
    else :
        raise ValueError('Unsupported data source. Use 12var or 29var')
    
    sequence_length = data.shape[1]
    print('Parameter Sequence Length:',sequence_length)
    number_of_features = data.shape[2]
    print('Parameter Number of Features:',number_of_features)
    Y_temporal=[len(x) for x in data]
    dataset=TensorDataset(torch.tensor(data).float())
    params['input_dim'] = number_of_features
    params['max_seq_len'] = sequence_length
    params['num_samples']=data.shape[0]
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=params['batch_size'],shuffle=True)
    