
import torch
import numpy as np
import wandb
from typing import Dict

def embedding_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    e_opt: torch.optim.Optimizer, 
    r_opt: torch.optim.Optimizer, 
    params: Dict
) -> None:
    """The training loop for the embedding and recovery functions using wandb."""
    for epoch in range(params['embedder_epoch']):
        for X_mb, T_mb in dataloader:
            model.zero_grad()
            _, E_loss0, E_loss_T0 = model(X=X_mb, T=T_mb, Z=None, obj="autoencoder")
            loss = np.sqrt(E_loss_T0.item())
            E_loss0.backward()
            e_opt.step()
            r_opt.step()

        with torch.no_grad():
                _, E_loss0_test, E_loss_T0_test = model(X=X_mb, T=T_mb, Z=None, obj="autoencoder")
                loss_test = np.sqrt(E_loss0_test.item())

        # Log metrics with wandb
        wandb.log({"train_loss": loss, "val_loss": loss_test}, step=epoch)
def supervisor_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    s_opt: torch.optim.Optimizer, 
    params: Dict
) -> None:
    """The training loop for the supervisor function using wandb."""
    for epoch in range(params['supervisor_epochs']):
        for X_mb, T_mb in dataloader:
            model.zero_grad()
            S_loss = model(X=X_mb, T=T_mb, Z=None, obj="supervisor")
            S_loss.backward()
            loss = np.sqrt(S_loss.item())
            s_opt.step()

        wandb.log({"supervisor_loss": loss}, step=epoch)
def joint_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    e_opt: torch.optim.Optimizer, 
    r_opt: torch.optim.Optimizer, 
    s_opt: torch.optim.Optimizer, 
    g_opt: torch.optim.Optimizer, 
    d_opt: torch.optim.Optimizer, 
    params: Dict
) -> None:
    """The joint training loop using wandb."""
    for epoch in range(params['supervisor_epochs']):
        for X_mb, T_mb in dataloader:
            # print('Shape of X_mb:', X_mb.shape)
            # print('Shape of T_mb:', T_mb.shape)
            Z_mb = torch.rand((params['batch_size'], params['max_seq_len'], params['Z_dim']))
            model.zero_grad()
            G_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="generator")
            G_loss.backward()
            g_opt.step()
            s_opt.step()

            model.zero_grad()
            E_loss, _, E_loss_T0 = model(X=X_mb, T=T_mb, Z=Z_mb, obj="autoencoder")
            E_loss.backward()
            e_opt.step()
            r_opt.step()

            model.zero_grad()
            D_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="discriminator")
            if D_loss > params['dis_thresh']:
                D_loss.backward()
                d_opt.step()

            wandb.log({"E_loss": E_loss.item(), "G_loss": G_loss.item(), "D_loss": D_loss.item()}, step=epoch)
