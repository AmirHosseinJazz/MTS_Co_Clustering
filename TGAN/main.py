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


def main(params):

    ### Loading the data - 12 variables or 29 variables
    if params["data_source"] == "29var":
        data = load_data(
            "../Data/PreProcessed/29var/df29.xlsx",
            break_to_smaller=params["break_data"],
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

    ###
    sequence_length = data.shape[1]
    print("Parameter Sequence Length:", sequence_length)
    number_of_features = data.shape[2]
    print("Parameter Number of Features:", number_of_features)
    Y_temporal = [len(x) for x in data]
    dataset = TimeGANDataset(data, Y_temporal)
    # Define hyperparameters and model configuration
    params["input_dim"] = number_of_features
    params["max_seq_len"] = sequence_length
    params["num_samples"] = data.shape[0]
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=params["batch_size"], shuffle=True
    )
    wandb.init(project="timegan", config=params)

    # Model and optimizers initialization
    device = params["device"]
    model = TimeGAN(params).to(device)
    e_opt = optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=0.005
    )
    r_opt = optim.Adam(list(model.recovery.parameters()), lr=0.005)
    s_opt = optim.Adam(list(model.supervisor.parameters()), lr=0.005)
    g_opt = optim.Adam(list(model.generator.parameters()), lr=0.005)
    d_opt = optim.Adam(list(model.discriminator.parameters()), lr=0.005)

    # Training loops
    embedding_trainer(
        model=model, dataloader=dataloader, e_opt=e_opt, r_opt=r_opt, params=params
    )

    supervisor_trainer(model=model, dataloader=dataloader, s_opt=s_opt, params=params)

    joint_trainer(
        model=model,
        dataloader=dataloader,
        e_opt=e_opt,
        r_opt=r_opt,
        s_opt=s_opt,
        g_opt=g_opt,
        d_opt=d_opt,
        params=params,
    )

    previous_models = os.listdir("saved_models")
    current_model_version = 1
    if len(previous_models) > 0:
        for prev_model in previous_models:
            current_model_version = current_model_version + 1
    current_model_name = "TimeGAN_model" + str(current_model_version)
    if not os.path.exists(f"./saved_models/{current_model_name}"):
        os.makedirs(f"./saved_models/{current_model_name}")
    torch.save(model.state_dict(), f"./saved_models/{current_model_name}/model.pth")
    ### Save config
    with open(f"./saved_models/{current_model_name}/config.txt", "w") as f:
        for key in params.keys():
            f.write(f"{key} : {params[key]}\n")
    # Finalize wandb
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Z_dim", type=int, default=50, help="Dimension of the random noise vector"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimension of the hidden layers in the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=15,
        help="Number of samples in each mini-batch",
    )
    parser.add_argument(
        "--dis_thresh",
        type=float,
        default=0.15,
        help="Threshold value for discriminator loss",
    )
    parser.add_argument(
        "--num_layers", type=int, default=5, help="Number of layers in the model"
    )
    parser.add_argument(
        "--embedder_epoch",
        type=int,
        default=500,
        help="Number of epochs for embedding training",
    )
    parser.add_argument(
        "--supervisor_epochs",
        type=int,
        default=500,
        help="Number of epochs for supervisor training",
    )
    parser.add_argument(
        "--padding_value", type=int, default=0, help="Value used for padding sequences"
    )
    parser.add_argument(
        "--module",
        type=str,
        default="GRU",
        help="Type of recurrent module used in the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (cpu or cuda)",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="29var",
        help="Data source to use (12var or 29var)",
    )
    parser.add_argument(
        "--break_data",
        type=bool,
        default=False,
        help="Break the data into smaller sequences",
    )
    parser.add_argument(
        "--break_size", type=int, default=60, help="Size of the smaller sequences"
    )
    parser.add_argument(
        "--leave_out_problematic_features",
        type=bool,
        default=True,
        help="Leave out the problematic features",
    )
    parser.add_argument(
        "--cutoff-data",
        type=bool,
        default=True,
        help="Cutoff data- it works only if break_data is False",
    )
    args = parser.parse_args()
    params = {
        "Z_dim": args.Z_dim,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "dis_thresh": args.dis_thresh,
        "num_layers": args.num_layers,
        "embedder_epoch": args.embedder_epoch,
        "supervisor_epochs": args.supervisor_epochs,
        "padding_value": args.padding_value,
        "module": args.module,
        "device": args.device,
        "data_source": args.data_source,
        "break_data": args.break_data,
        "break_size": args.break_size,
        "leave_out_problematic_features": args.leave_out_problematic_features,
        "cutoff_data": args.cutoff_data,
    }
    main(params)
