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
    
    