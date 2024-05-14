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
    # sns.distplot(lengths)
    print('Number of samples:',len(samples))
    print('Number of Names:',len(DF.Name.unique().tolist()))

    all_samples=pd.concat(samples)
    np_samples=[]
    for k in all_samples.Name.unique().tolist():
        temp_df=all_samples[all_samples['Name']==k]
        temp_df=temp_df.drop('Name',axis=1)
        ### Filling NaN values with mean of the column
        temp_df=temp_df.fillna(temp_df.mean())
        np_samples.append(np.array(temp_df))
    np_samples=np.array(np_samples)
    print('Shape of np_samples:',np_samples.shape)
    return np_samples


