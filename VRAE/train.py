import utils
from model import VRAE
import numpy as np
import torch

import plotly
from torch.utils.data import DataLoader, TensorDataset
import wandb

def model_pipeline(hyperparameters):

	# tell wandb to get started
	with wandb.init(project="VRAE", config=hyperparameters):
		# access all HPs through wandb.config, so logging matches execution!
		config = wandb.config

		# Initialize VRAE object
		model = VRAE(sequence_length=config.sequence_length,
					number_of_features=config.number_of_features,
					hidden_size=config.hidden_size,
					hidden_layer_depth=config.hidden_layer_depth,
					latent_length=config.latent_length,
					batch_size=config.batch_size,
					learning_rate=config.learning_rate,
					epochs=config.epochs,
					dropout_rate=config.dropout_rate,
					optimizer=config.optimizer,
					clip=config.clip,
					max_grad_norm=config.max_grad_norm,
					criterion=config.criterion,
					block=config.block,
					cuda=True,
					print_every=30,
					dload='./saved')

		# and use them to train the model
		train(model, config)

	return model


def train(model, config):
	# Tell wandb to watch what the model gets up to: gradients, weights, and more!
	criterion = model.get_criterion()
	wandb.watch(model, criterion, log='all', log_freq=10)

	# Dataloader
	train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
	valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, drop_last=True)

	# Fit & transform
	model.fit(train_loader, valid_loader, save=True)

if __name__ == '__main__':
	data=utils.load_data('../../NSMD/PreProcessed/29var/df29.xlsx')
	X_train,X_val=utils.split_train_val(data)
	train_dataset = TensorDataset(torch.from_numpy(X_train))
	valid_dataset = TensorDataset(torch.from_numpy(X_val))
	sequence_length = X_train.shape[1]
	number_of_features = X_train.shape[2]
	config = dict(
		dataset=data,
		sequence_length=sequence_length,
		number_of_features=number_of_features,
		hidden_size=120,
		hidden_layer_depth=2, # 1, 2
		latent_length=16,
		batch_size=4,
		block='LSTM', # LSTM, GRU
		epochs=120,
		dropout_rate=0.2,
		optimizer='Adam', # Adam, SGD
		learning_rate=0.0005,
		criterion='MSELoss', # SmoothL1Loss, MSELoss
		clip=True,
		max_grad_norm=5,
	)
	model = model_pipeline(config)