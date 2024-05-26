import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import wandb
from util import load_data
from dataset import TimeVAEdataset
from TimeVAE import TimeVAE
import argparse


def main(params):
    ### Loading Data
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

    params["feat_dim"] = number_of_features
    params["seq_len"] = sequence_length
    params["num_samples"] = data.shape[0]
    params["trend_poly"] = 2
    params["custom_seas"] = [(4, 8)]
    params["reconstruction_wt"] = 3.0
    params["hidden_layer_sizes"] = [50, 100, 200]

    dataset = TimeVAEdataset(data)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=params["batch_size"], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=params["batch_size"], shuffle=True
    )
    # wandb.init(project="timevae", config=params)
    ### Model and optimizers initialization and training

    model = TimeVAE(params).to(params["device"])
    model.fit(
        train_loader,
        test_loader,
        epochs=params["epochs"],
        lr=params["lr"],
    )

    # --Saving Model
    previous_models = os.listdir("saved_models")
    current_model_version = 1
    if len(previous_models) > 0:
        for prev_model in previous_models:
            current_model_version = current_model_version + 1
    current_model_name = "TimeVAE_model" + str(current_model_version)
    if not os.path.exists(f"./saved_models/{current_model_name}"):
        os.makedirs(f"./saved_models/{current_model_name}")
    torch.save(model.state_dict(), f"./saved_models/{current_model_name}/model.pth")
    ### Save config
    with open(f"./saved_models/{current_model_name}/config.txt", "w") as f:
        for key in params.keys():
            f.write(f"{key} : {params[key]}\n")
    # Finalize wandb
    # wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimeVAE")
    parser.add_argument(
        "--data_source",
        type=str,
        default="29var",
        help="Data source to use for training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=20, help="Batch size for training"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=64, help="Hidden dimension for the model"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for training"
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
        "data_source": args.data_source,
        "batch_size": args.batch_size,
        "device": args.device,
        "epochs": args.epochs,
        "latent_dim": args.latent_dim,
        "lr": args.lr,
        "break_data": args.break_data,
        "break_size": args.break_size,
        "leave_out_problematic_features": args.leave_out_problematic_features,
        "cutoff_data": args.cutoff_data,
    }
    main(params)
