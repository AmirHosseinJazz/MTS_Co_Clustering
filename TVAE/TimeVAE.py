import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb


class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.seq_len = params["seq_len"]
        self.feat_dim = params["feat_dim"]
        self.latent_dim = params["latent_dim"]
        self.device = params["device"]
        # Define convolutional layers
        modules = []
        in_channels = params["feat_dim"]
        for h_dim in params["hidden_layer_sizes"]:
            modules.append(
                nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=2, padding=1)
            )
            modules.append(nn.ReLU())
            in_channels = h_dim

        self.conv_layers = nn.Sequential(*modules)
        self.flatten = nn.Flatten()
        self.fc_z_mean = nn.Linear(
            params["hidden_layer_sizes"][-1]
            * (self.seq_len // 2 ** len(params["hidden_layer_sizes"])),
            self.latent_dim,
        )
        self.fc_z_log_var = nn.Linear(
            params["hidden_layer_sizes"][-1]
            * (self.seq_len // 2 ** len(params["hidden_layer_sizes"])),
            self.latent_dim,
        )
        self.sampling = Sampling()

    def forward(self, x):
        x = x.to_device(self.device)
        x = x.permute(
            0, 2, 1
        )  # Change from (batch, seq_len, feat_dim) to (batch, feat_dim, seq_len)
        x = self.conv_layers(x)
        x = self.flatten(x)
        z_mean = self.fc_z_mean(x)
        z_log_var = self.fc_z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class TrendLayer(nn.Module):
    def __init__(self, params):
        super(TrendLayer, self).__init__()
        self.latent_dim = params["latent_dim"]
        self.feat_dim = params["feat_dim"]
        self.trend_poly = params["trend_poly"]
        self.seq_len = params["seq_len"]
        self.trend_dense1 = nn.Linear(
            params["latent_dim"], params["feat_dim"] * params["trend_poly"]
        )
        self.trend_dense2 = nn.Linear(
            params["feat_dim"] * params["trend_poly"],
            params["feat_dim"] * params["trend_poly"],
        )
        self.device = params["device"]

    def forward(self, z):
        # Process the input through two dense layers
        # print("Shape of trend layer:", z.shape)
        z = z.to(self.device)
        trend_params = F.relu(self.trend_dense1(z))
        # print("Shape of trend_params after first dense layer:", trend_params.shape)
        trend_params = self.trend_dense2(trend_params)
        # print("Shape of trend_params after second dense layer:", trend_params.shape)
        trend_params = trend_params.view(-1, self.feat_dim, self.trend_poly)
        # print("Shape of trend_params after reshaping:", trend_params.shape)
        # Create the polynomial terms for the trend
        lin_space = torch.linspace(0, 1, self.seq_len, device=z.device)
        poly_space = torch.stack(
            [lin_space ** (p + 1) for p in range(self.trend_poly)], dim=0
        )

        # Calculate the trend values
        trend_vals = torch.matmul(trend_params, poly_space)
        trend_vals = trend_vals.permute(0, 2, 1)

        return trend_vals


class SeasonalLayer(nn.Module):
    def __init__(self, params):
        super(SeasonalLayer, self).__init__()
        self.feat_dim = params["feat_dim"]
        self.seq_len = params["seq_len"]
        self.latent_dim = params["latent_dim"]
        self.custom_seas = params["custom_seas"]
        self.device = params["device"]
        self.dense_layers = nn.ModuleList(
            [
                nn.Linear(self.latent_dim, self.feat_dim * num_seasons)
                for num_seasons, _ in self.custom_seas
            ]
        )
        self.reshape_layers = [
            (self.feat_dim, num_seasons) for num_seasons, _ in self.custom_seas
        ]

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = torch.arange(num_seasons).repeat(len_per_season)
        season_indexes = season_indexes.unsqueeze(0).repeat(
            self.seq_len // len_per_season + 1, 1
        )
        season_indexes = season_indexes.flatten()[: self.seq_len]
        return season_indexes

    def forward(self, z):
        all_seas_vals = []

        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)
            season_params = season_params.view(-1, self.feat_dim, num_seasons)

            season_indexes = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            )
            season_vals = F.embedding(season_indexes, season_params.permute(0, 2, 1))
            all_seas_vals.append(season_vals)

        all_seas_vals = torch.stack(all_seas_vals, dim=-1)
        all_seas_vals = torch.sum(all_seas_vals, dim=-1)
        return all_seas_vals


class LevelModel(nn.Module):
    def __init__(self, params):
        super(LevelModel, self).__init__()
        self.feat_dim = params["feat_dim"]
        self.seq_len = params["seq_len"]
        self.latent_dim = params["latent_dim"]
        self.device = params["device"]
        self.level_dense1 = nn.Linear(self.latent_dim, self.feat_dim)
        self.level_dense2 = nn.Linear(self.feat_dim, self.feat_dim)

    def forward(self, z):
        # Process input through two dense layers
        z = z.to(self.device)
        level_params = F.relu(self.level_dense1(z))
        level_params = self.level_dense2(level_params)

        # Reshape to (N, 1, D)
        level_params = level_params.view(-1, 1, self.feat_dim)

        # Create a tensor of ones with shape (1, seq_len, 1) and broadcast it
        ones_tensor = torch.ones(
            (1, self.seq_len, 1), device=z.device, dtype=torch.float32
        )

        # Multiply level_params by ones_tensor to broadcast the level values across the sequence length
        level_vals = level_params * ones_tensor

        return level_vals


class DecoderResidual(nn.Module):
    def __init__(self, params):
        super(DecoderResidual, self).__init__()
        self.input_dim = params["input_dim"]
        self.hidden_layer_sizes = params["hidden_layer_sizes"]
        self.seq_len = params["seq_len"]
        self.feat_dim = params["feat_dim"]
        self.device = params["device"]
        # Dense layer to reshape input to suitable dimension
        self.dense = nn.Linear(self.input_dim, self.hidden_layer_sizes[-1] * self.seq_len)

        # Convolutional transpose layers
        self.conv_transpose_layers = nn.ModuleList()
        # Initialize convolutional transpose layers for upsampling
        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            self.conv_transpose_layers.append(
            nn.ConvTranspose1d(
                in_channels=self.hidden_layer_sizes[-i - 1],
                out_channels=num_filters,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
            )

        # Last convolutional transpose to match feature dimensions
        self.final_conv_transpose = nn.ConvTranspose1d(
            in_channels=self.hidden_layer_sizes[0],
            out_channels=self.feat_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

    def forward(self, x):
        x= x.to(self.device)
        x = F.relu(self.dense(x))
        x = x.view(-1, self.hidden_layer_sizes[-1], self.seq_len)

        for conv_layer in self.conv_transpose_layers:
            x = F.relu(conv_layer(x))

        x = F.relu(self.final_conv_transpose(x))
        x = x.view(-1, self.seq_len, self.feat_dim)
        return x


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.latent_dim = params["latent_dim"]
        self.feat_dim = params["feat_dim"]
        self.seq_len = params["seq_len"]
        self.trend_poly = params["trend_poly"]
        self.custom_seas = params["custom_seas"]
        self.device = params["device"]
        # Initialize components
        if self.trend_poly > 0:
            self.trend_layer = TrendLayer(params)
        if len(self.custom_seas) > 0:
            self.seasonal_layer = SeasonalLayer(params)
        self.level_layer = LevelModel(params)
        # Residual connection
        self.residual_dense = nn.Linear(self.latent_dim, self.seq_len * self.feat_dim)
        self.use_residual_conn = True

    def forward(self, z):
        z= z.to(self.device)
        outputs = self.level_layer(z)
        # print("Shape of outputs after level layer:", outputs.shape)
        if self.trend_poly > 0:
            trend_vals = self.trend_layer(z)
            outputs = trend_vals if outputs is None else outputs + trend_vals
        # print("Shape of outputs after trend layer:", outputs.shape)
        # if len(self.custom_seas) > 0:
        #     seasonal_vals = self.seasonal_layer(z)
        #     outputs = seasonal_vals if outputs is None else outputs + seasonal_vals

        if self.use_residual_conn:
            residuals = self.residual_dense(z)
            residuals = residuals.view(-1, self.seq_len, self.feat_dim)
            outputs = residuals if outputs is None else outputs + residuals

        return outputs


class TimeVAE(nn.Module):
    def __init__(
        self,
        params,
    ):
        super(TimeVAE, self).__init__()
        self.seq_len = params["seq_len"]
        self.feat_dim = params["feat_dim"]
        self.latent_dim = params["latent_dim"]
        self.reconstruction_wt = params["reconstruction_wt"]
        self.batch_size = params["batch_size"]
        self.hidden_layer_sizes = params["hidden_layer_sizes"]
        self.trend_poly = params["trend_poly"]
        self.custom_seas = params["custom_seas"]
        self.device = params["device"]
        # Define Encoder and Decoder
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        x=x.to(self.device)
        z_mean, z_log_var, _ = self.encoder(x)
        z = self._reparameterize(z_mean, z_log_var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, z_mean, z_log_var

    def _reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def loss_function(self, recon_x, x, mean, log_var):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # Total loss
        return self.reconstruction_wt * recon_loss + kl_loss

    def train_step(self, data, optimizer):
        self.train()
        optimizer.zero_grad()
        recon_batch, mu, logvar = self(data)
        loss = self.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        return loss.item()

    def test_step(self, data):
        self.eval()
        with torch.no_grad():
            recon_batch, mu, logvar = self(data)
            test_loss = self.loss_function(recon_batch, data, mu, logvar).item()
        return test_loss

    def fit(self, train_loader, test_loader, epochs=20, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            train_loss = 0.0
            for batch in train_loader:
                loss = self.train_step(batch, optimizer)
                train_loss += loss
            train_loss /= len(train_loader)

            test_loss = 0.0
            for batch in test_loader:
                loss = self.test_step(batch)
                test_loss += loss
            test_loss /= len(test_loader)

            print(
                f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
            )
        # wandb.log({"train_loss": train_loss, "test_loss": test_loss})
