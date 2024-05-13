import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import wandb
from util import load_data
from TimeGAN import TimeGAN
from train import embedding_trainer, supervisor_trainer, joint_trainer
from dataset import TimeGANDataset
import argparse

def generate_samples(model_name, params):
    """Generate samples from the trained TimeGAN model."""
    # Load the model
    model = TimeGAN(params)
    model.load_state_dict(torch.load(f"saved_models/{model_name}/model.pth"))
    model.eval()

    # Generate a 1-Dvector of random noise
    Z = torch.randn(params['batch_size'],params['max_seq_len'], params['Z_dim'])

    # Generate samples using the generator
    with torch.no_grad():
        samples = model.forward(X=None,T=params['max_seq_len'],Z=Z, obj="inference")
    
    # Store samples
    if not os.path.exists(f'../Generated/{model_name}/'):
        os.makedirs(f'../Generated/{model_name}/')

    #print the generated samples
    print(samples.shape)
    print(samples[0])
    # Save the generated samples
    np.save(f'../Generated/{model_name}/generated_samples.npy', samples)

def encode(data, model_name, params):
    """Encode the data using the trained TimeGAN model."""
    # Load the model
    model = TimeGAN(params)
    model.load_state_dict(torch.load(f"saved_models/{model_name}/model.pth"))
    model.eval()

    # Create a DataLoader
    Y_temporal=[len(x) for x in data]
    dataset = TimeGANDataset(data, Y_temporal)
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False)

    # Encode the data
    encoded_data = []
    with torch.no_grad():
        for X_mb, T_mb in dataloader:
            encoded_data.append(model(X=X_mb, T=T_mb, Z=None, obj="embeder_out"))
    # Store the encoded data
    if not os.path.exists(f'../Generated/{model_name}/'):
        os.makedirs(f'../Generated/{model_name}/')

    # print the encoded data
    final_encoded=np.concatenate(encoded_data, axis=0)
    print(final_encoded.shape)

    np.save(f'../Generated/{model_name}/encoded_data.npy',final_encoded )



def main():
    parser = argparse.ArgumentParser(description='TimeGAN Inference')
    parser.add_argument('--model_name', type=str, default="TimeGAN_model1", help='Name of the trained TimeGAN model')
    parser.add_argument('--objective', type=str,default='generate', help='Objective for encoding (embeder_out or supervisor_out)')
    parser.add_argument('--data_source',type=str ,default='12var',help='Data source to use (12var or 29var)')
    args = parser.parse_args()


    model_name=args.model_name
    params = {}
    try:
        with open(f'./saved_models/{model_name}/config.txt', 'r') as f:
            for line in f:
                key, value = line.split(':')
                # check if the parameter is an integer
                try:
                    params[key.strip()] = int(value.strip())
                except:
                    params[key.strip()] = value.strip()
    except Exception as E:
        raise Exception(f"Could not load the configuration file: {E}")
    # print('Parameters:', params)
    if args.objective == 'generate':
        generate_samples(args.model_name, params)
    elif args.objective == 'encode':
        if args.data_source=='29var':
            data=load_data('../Data/PreProcessed/29var/df29.xlsx')
        elif args.data_source=='12var':
            data=load_data('../Data/PreProcessed/12var/df12.xlsx')
        else :
            raise ValueError('Unsupported data source. Use 12var or 29var')
        encode(data, args.model_name, params)

if __name__ == '__main__':
    main()