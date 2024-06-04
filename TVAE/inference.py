import os
import argparse
import torch
import torch.optim as optim
import wandb
import numpy as np
from util import load_data
from TimeVAE import TimeVAE
from dataset import TimeVAEdataset

import ast


def generate_samples(model_name, params):
    """
    Generate samples from the trained TimeVAE model.
    """
    model = TimeVAE(params).to(params["device"])
    model.load_state_dict(torch.load(f"./saved_models/{model_name}/model.pth"))
    model.eval()
    generated_samples = []
    for _ in range(180):
        z = torch.randn(params["latent_dim"]).to(params["device"])
        with torch.no_grad():
            samples = model.decoder(z)
        samples = samples.cpu()
        generated_samples.append(samples)
    generated_samples = torch.cat(generated_samples, dim=0)

    if not os.path.exists(f"../Generated/{model_name}/"):
        os.makedirs(f"../Generated/{model_name}/")
    print(generated_samples.shape)
    # print(generated_samples[0])
    print(model_name)

    np.save(
        f"../Generated/{model_name}/generated_samples.npy", generated_samples.numpy()
    )


def encode(data, model_name, params):
    """
    Encode the data using the trained TimeVAE model.
    """
    model = TimeVAE(params)
    model.load_state_dict(torch.load(f"saved_models/{model_name}/model.pth"))
    model.to(params['device'])
    model.eval()
    print('Data Shape is',data.shape)

    dataset = TimeVAEdataset(data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=params["batch_size"], shuffle=False
    )

    encoded_data = []

    ### Change this to use the model
    with torch.no_grad():
        for X_mb in dataloader:
            X_mb = X_mb.to(params["device"]) 
            encoded_output = model.encoder(X_mb)[-1]
            encoded_output = encoded_output.cpu()  # Move the encoded output back to CPU
            encoded_data.append(encoded_output)
    encoded_data = torch.cat(encoded_data, dim=0)
    print(encoded_data.shape)
    if not os.path.exists(f"../Generated/{model_name}/"):
        os.makedirs(f"../Generated/{model_name}/")
    np.save(f"../Generated/{model_name}/encoded_data.npy", encoded_data.numpy())


def main():
    parser = argparse.ArgumentParser(description="TimeVAE Inference")
    parser.add_argument(
        "--model_name",
        type=str,
        default="TimeVAE_model1",
        help="Name of the trained TimeVAE model",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="generate",
        help="Objective for encoding (embeder_out or supervisor_out)",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="12var",
        help="Data source to use (12var or 29var)",
    )
    args = parser.parse_args()

    model_name = args.model_name
    params = {}
    try:
        with open(f"./saved_models/{model_name}/config.txt", "r") as f:
            for line in f:
                key, value = line.split(":")
                # check if the parameter is an integer
                try:
                    if key.strip() in ["hidden_layer_sizes"]:
                        # params[key.strip()] = value.strip().split(',')
                        # print(value)
                        params[key.strip()] = [
                            int(x.strip())
                            for x in value.replace("[", "")
                            .replace("]", "")
                            .strip()
                            .split(",")
                        ]
                    else:
                        params[key.strip()] = int(value.strip())

                except:
                    params[key.strip()] = value.strip()
    except Exception as E:
        raise Exception(f"Could not load the configuration file: {E}")
    params["custom_seas"] = [(4, 8)]

    if args.objective == "generate":
        generate_samples(args.model_name, params)
    elif args.objective == "encode":
        if params["data_source"] == "29var":
            data = load_data(
                "../Data/PreProcessed/29var/df29.xlsx",
                break_to_smaller=False,
                break_size=params["break_size"],
                leave_out_problematic_features=params["leave_out_problematic_features"],
                cutoff_data=params["cutoff_data"],
            )
        elif params["data_source"] == "12var":
            data = load_data(
                "../Data/PreProcessed/12var/df12.xlsx",
                break_to_smaller=params["break_data"],
                break_size=params["break_size"],
                leave_out_problematic_features=params["leave_out_problematic_features"],
                cutoff_data=params["cutoff_data"],
            )
        else:
            raise ValueError("Unsupported data source. Use 12var or 29var")
        encode(data, args.model_name, params)


if __name__ == "__main__":
    main()
