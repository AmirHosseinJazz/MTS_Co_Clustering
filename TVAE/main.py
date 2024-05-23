import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import wandb
from util import load_data
from dataset import TimeVAEDataset
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
    dataset=TimeVAEDataset(data,Y_temporal)
    params['input_dim'] = number_of_features
    params['max_seq_len'] = sequence_length
    params['num_samples']=data.shape[0]
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=params['batch_size'],shuffle=True)
    wandb.init(project="timevae", config=params)

    ### Model and optimizers initialization and training
    device = params['device']
    
    #
    previous_models=os.listdir('saved_models')
    current_model_version=1
    if len(previous_models)>0:
        for prev_model in previous_models:
            current_model_version=current_model_version+1
    current_model_name='TimeVAE_model'+str(current_model_version)
    if not os.path.exists(f'./saved_models/{current_model_name}'):
        os.makedirs(f'./saved_models/{current_model_name}')
    torch.save(model.state_dict(), f'./saved_models/{current_model_name}/model.pth')
    ### Save config
    with open(f'./saved_models/{current_model_name}/config.txt','w') as f:
        for key in params.keys():
            f.write(f'{key} : {params[key]}\n')
    # Finalize wandb
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimeVAE')
    parser.add_argument('--data_source', type=str, default='29var', help='Data source to use for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for the model')
    
    args = parser.parse_args()
    params = {
        'data_source': args.data_source,
        'batch_size': args.batch_size,
        'device': args.device,
        'epochs': args.epochs,
        'hidden_dim': args.hidden_dim
    }
    main(params)
