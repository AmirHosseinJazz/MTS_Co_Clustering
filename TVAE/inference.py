import os 
import argparse
import torch
import torch.optim as optim
import wandb
import numpy as np
from util import load_data
from TimeVAE import TimeVAE
from dataset import TimeVAEDataset

def generate_samples(model_name,params):
    """
    Generate samples from the trained TimeVAE model.
    """
    model=TimeVAE(params)
    model.load_state_dict(torch.load(f"./saved_models/{model_name}/model.pth"))
    model.eval()

    generated_samples=[]
    for _ in range(10):
        Z=torch.randn(params['batch_size'],params['max_seq_len'],params['Z_dim'])
        with torch.no_grad():
            samples=model.forward(X=None,T=params['max_seq_len'],Z=Z,obj="inference")
        samples=samples.cpu()
        generated_samples.append(samples)
    generated_samples=torch.cat(generated_samples,dim=0)

    if not os.path.exists(f'../Generated/{model_name}/'):
        os.makedirs(f'../Generated/{model_name}/')
    print(generated_samples.shape)
    print(generated_samples[0])
    print(model_name)

    np.save(f'../Generated/{model_name}/generated_samples.npy',generated_samples.numpy())

def encode(data,model_name,params):
    """
    Encode the data using the trained TimeVAE model.
    """
    model=TimeVAE(params)
    model.load_state_dict(torch.load(f"saved_models/{model_name}/model.pth"))
    model.eval()

    Y_temporal=[len(x) for x in data]
    dataset=TimeVAEDataset(data,Y_temporal)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=params['batch_size'],shuffle=False)

    encoded_data=[]

    ### Change this to use the model
    # with torch.no_grad():
    #     for X_mb,T_mb in dataloader:
    #         X_mb=torch.tensor(X_mb,dtype=torch.float32,device=params['device'])
    #         T_mb=torch.tensor(T_mb,dtype=torch.float32,device=params['device'])
    #         Z_mb=torch.randn((params['batch_size'],params['max_seq_len'],params['Z_dim'])).to(params['device'])
    #         encoded_data.append(model(X=X_mb,T=T_mb,Z=Z_mb,obj="encoder"))
    # encoded_data=torch.cat(encoded_data,dim=0)
    
    if not os.path.exists(f'../Generated/{model_name}/'):
        os.makedirs(f'../Generated/{model_name}/')
    np.save(f'../Generated/{model_name}/encoded_data.npy',encoded_data.numpy())

def main():
    parser=argparse.ArgumentParser(description='TimeVAE Inference')
    parser.add_argument('--model_name', type=str, default="TimeVAE_model1", help='Name of the trained TimeVAE model')
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
    