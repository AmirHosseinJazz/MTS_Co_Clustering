import os
import numpy as np
import pandas as pd
from random import randint
###
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = [20, 4]
import seaborn as sns
###
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(file_path):
    DF=pd.read_excel(file_path)
    DF['datetime']=DF['Date']+' '+DF['Time']
    DF['datetime']=pd.to_datetime(DF['datetime'])
    DF.set_index('datetime',inplace=True)
    DF.drop(['Date','Time','Duration'],axis=1,inplace=True)
    DF.sort_index(inplace=True)
    #

    print('Number of unique Names:',DF['Name'].nunique())
    print('Number of columns:',len(DF.columns))
    print('Number of rows:',len(DF))
    print('Name of columns:',DF.columns)
    
    lengths=[]
    samples=[]
    for k in DF.Name.unique().tolist():
        lengths.append(len(DF[DF['Name']==k]))
        if len(DF[DF['Name']==k])>=120:
            temp=DF[DF['Name']==k]
            samples.append(temp.iloc[:120,:])
    sns.distplot(lengths)
    print('Number of samples:',len(samples))
    print('Number of Names:',len(DF.Name.unique().tolist()))

    samples=pd.concat(samples)
    np_samples=[]
    for k in samples.Name.unique().tolist():
        np_samples.append(np.array(samples[samples['Name']==k]))
    np_samples=np.array(np_samples)
    print('Shape of np_samples:',np_samples.shape)
    return np_samples

def split_train_val(data,ratio=0.7):
    N, D, _ = data.shape  # Get the shape of the data

    ind_cut = int(ratio * N)
    ind = np.random.RandomState(seed=42).permutation(N)

    # Split data into training and validation sets
    X_train = data[ind[:ind_cut]]  # Select training data
    X_val = data[ind[ind_cut:]]    # Select validation data
    return X_train,X_val

