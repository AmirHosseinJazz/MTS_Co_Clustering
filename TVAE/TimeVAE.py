import torch
import torch.nn as nn
import torch.nn.functional as F

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std
class BaseVariationalAutoencoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.seq_len = params['seq_len']
        self.feat_dim = params['feat_dim']
        self.latent_dim = params['latent_dim']
        self.reconstruction_wt = params['reconstruction_wt']
        self.encoder = Encoder(params) 
        self.decoder = Decoder(params) 

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        if self.decoder:
            reconstruction = self.decoder(z)
            return reconstruction, z_mean, z_log_var
        return z_mean, z_log_var

    def loss_function(self, x, reconstruction, z_mean, z_log_var):
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return self.reconstruction_wt * recon_loss + kl_loss

class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.seq_len = params['seq_len']
        self.feat_dim = params['feat_dim']
        self.latent_dim = params['latent_dim']

        # Define convolutional layers
        modules = []
        in_channels = self.feat_dim
        for h_dim in params['hidden_layer_sizes']:
            modules.append(nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=2, padding=1))
            modules.append(nn.ReLU())
            in_channels = h_dim

        self.conv_layers = nn.Sequential(*modules)
        self.flatten = nn.Flatten()
        self.fc_z_mean = nn.Linear(params['hidden_layer_sizes'][-1] * (self.seq_len // 2 ** len(params['hidden_layer_sizes'])), self.latent_dim)
        self.fc_z_log_var = nn.Linear(params['hidden_layer_sizes'][-1] * (self.seq_len // 2 ** len(params['hidden_layer_sizes'])), self.latent_dim)
        self.sampling = Sampling()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change from (batch, seq_len, feat_dim) to (batch, feat_dim, seq_len)
        x = self.conv_layers(x)
        x = self.flatten(x)
        z_mean = self.fc_z_mean(x)
        z_log_var = self.fc_z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z
class TrendModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.seq_len = params['seq_len']
        self.feat_dim = params['feat_dim']
        self.trend_poly = params['trend_poly']

        # Polynomial coefficients
        self.coefficients = nn.Parameter(torch.randn(self.feat_dim, self.trend_poly))

    def forward(self, x):
        # Generate polynomial terms
        t = torch.linspace(0, 1, steps=self.seq_len, device=x.device)
        T = torch.stack([t**i for i in range(1, self.trend_poly + 1)], dim=1)
        
        # Compute polynomial trend for each feature
        trend = torch.einsum('bdi,ip->bpd', self.coefficients.expand(x.size(0), -1, -1), T)
        return trend.permute(0, 2, 1)

class SeasonalModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.seq_len = params['seq_len']
        self.num_seasons = params['num_seasons']
        self.amplitudes = nn.Parameter(torch.randn(params['feat_dim'], params['num_seasons']))
        self.frequencies = nn.Parameter(torch.randn(params['feat_dim'], params['num_seasons']))
        self.phases = nn.Parameter(torch.randn(params['feat_dim'], params['num_seasons']))

    def forward(self, x):
        t = torch.linspace(0, 1, steps=self.seq_len, device=x.device)
        freqs = torch.exp(self.frequencies)  # Ensure positive frequencies
        phases = self.phases
        sines = torch.sin(2 * np.pi * freqs.unsqueeze(-1) * t + phases.unsqueeze(-1))
        seasonal = torch.einsum('bdi,dj->bdj', self.amplitudes.expand(x.size(0), -1, -1), sines)
        return seasonal.permute(0, 2, 1)

class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.seq_len = params['seq_len']
        self.feat_dim = params['feat_dim']
        self.latent_dim = params['latent_dim']
        self.hidden_sizes = params['hidden_sizes']
        self.trend_poly = params['trend_poly']
        self.num_seasons = params['num_seasons']

        # Define layers
        self.fc1 = nn.Linear(self.latent_dim, self.hidden_sizes[-1])
        self.trend_model = TrendModel(self.seq_len, self.feat_dim, self.trend_poly)
        self.seasonal_model = SeasonalModel(self.seq_len, self.feat_dim, self.num_seasons)
        
        # Layers for the final output generation
        self.conv_transpose_layers = nn.ModuleList()
        current_channels = self.hidden_sizes[-1]
        for h_dim in reversed(self.hidden_sizes[:-1]):
            self.conv_transpose_layers.append(
                nn.ConvTranspose1d(current_channels, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            self.conv_transpose_layers.append(nn.ReLU())
            current_channels = h_dim

        self.final_layer = nn.ConvTranspose1d(current_channels, self.feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1)  # Add a fake sequence dimension

        # Apply convolution transpose layers
        for layer in self.conv_transpose_layers:
            x = layer(x)

        x = self.final_layer(x).squeeze(-1)

        # Add trend and seasonal components
        trend = self.trend_model(x)
        seasonal = self.seasonal_model(x)
        x = x + trend + seasonal

        return x
    def save_model(self, path):
        torch.save(self.state_dict(), './model.pth')
        # savve dictionary to txt file
        with open(path + '.txt', 'w') as file:
            file.write(str(self.__dict__))