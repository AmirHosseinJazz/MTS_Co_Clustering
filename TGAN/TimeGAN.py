import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self,param):
        super(Embedder, self).__init__()
        self.input_dim=param['input_dim'] # Number of features =12 or 29
        self.hidden_dim=param['hidden_dim'] # Hyperparameter
        self.num_layers=param['num_layers'] # Hyperparameter
        self.module=param['module'] # LSTM or GRU
        self.max_seq_len=param['max_seq_len'] # Maximum of sequence length
        self.padding_value=param['padding_value'] # Padding value
        self.device=param['device']
        if self.module.lower() == 'lstm':
            self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True).to(self.device)
        elif self.module.lower() == 'gru':
            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True).to(self.device)
        else:
            raise ValueError('Only support LSTM and GRU')
        
        self.fc=nn.Linear(self.hidden_dim,self.hidden_dim).to(self.device)
        self.act=nn.Sigmoid()
    
    def forward(self, x,seq_lengths):
        x = x.to(self.device)  # Ensure input tensor is on the correct device
        seq_lengths = seq_lengths.to('cpu')
        # seq_lengths is a list of lengths of each sequence in the batch [120,100,....]
        packed_input= nn.utils.rnn.pack_padded_sequence(x,seq_lengths,batch_first=True,enforce_sorted=False)
        packed_output, _= self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        output=self.act(self.fc(output))
        # Shape should be (batch_size, seq_len, hidden_dim)

        return output
class Generator(nn.Module):
    def __init__(self, param):
        super(Generator, self).__init__()
        self.num_layers = param['num_layers'] # Hyperparameter
        self.hidden_dim = param['hidden_dim'] # Hyperparameter
        self.module = param['module'] # LSTM or GRU
        
        # self.max_seq_len = param['max_seq_len'] # Maximum of sequence length e.g. 200
        self.input_dim = param['input_dim'] # Number of features e.g. 12 or 29
        self.z_dim = param['Z_dim'] # Dimension of noise
        self.device=param['device']
        if self.module.lower()=='lstm':
            self.rnn = nn.LSTM(self.z_dim, self.hidden_dim, self.num_layers, batch_first=True).to(self.device)
        elif self.module.lower()=='gru':
            self.rnn = nn.GRU(self.z_dim, self.hidden_dim, self.num_layers, batch_first=True).to(self.device)
        else:
            raise ValueError('Only support LSTM and GRU')
        
        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)  # Output to input_dim
        self.act = nn.Sigmoid()
        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.fc.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, Z, T=None):
        # T=T.to(self.device)
        Z=Z.to(self.device)
        # Z = (Batch, Seq_len, Z_dim)
        outpuy, _ = self.rnn(Z)
        linear = self.fc(outpuy)
        generated = self.act(linear)
        # Shape should be (batch_size, seq_len, hidden_dim)
        return generated
class Discriminator(nn.Module):
    def __init__(self, param):
        super(Discriminator, self).__init__()
        self.hidden_dim = param['hidden_dim'] # Hyperparameter
        self.num_layers = param['num_layers'] # Hyperparameter
        self.module=param['module'] # LSTM or GRU
        self.device=param['device']
        if self.module.lower()=='lstm':
            self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True).to(self.device)
        elif self.module.lower()=='gru':
            self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True).to(self.device)
        else:
            raise ValueError('Only support LSTM and GRU')
        self.linear = nn.Linear(self.hidden_dim, 1).to(self.device)


        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
    def forward(self, x, T=None):
        x=x.to(self.device)
        T=T.to(self.device)
        output, _ = self.rnn(x)
        linear = self.linear(output)
        discriminated = linear.squeeze(-1)
        # Shape should be (batch_size, seq_len)
        return discriminated
class Recovery(nn.Module):
    def __init__(self, param):
        super(Recovery, self).__init__()
        self.hidden_dim = param['hidden_dim']
        self.input_dim = param['input_dim']
        self.num_layers = param['num_layers']
        self.padding_value = param['padding_value']
        self.max_seq_len = param['max_seq_len']
        self.module = param['module']
        self.device=param['device']
        if self.module.lower()=='lstm':
            self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True).to(self.device)
        elif self.module.lower()=='gru':
            self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True).to(self.device)
        else:
            raise ValueError('Only support LSTM and GRU')
        self.fc = nn.Linear(self.hidden_dim, self.input_dim).to(self.device)

        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.fc.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
    def forward(self, x,T=None):
        x=x.to(self.device)
        # T=T.to(self.device)
        
        output, _ = self.rnn(x)
        linear_recovered = self.fc(output)
        # Shape should be (batch_size, seq_len, input_dim)
        return linear_recovered
class Supervisor(nn.Module):
    def __init__(self, param):
        super(Supervisor, self).__init__()
        self.device=param['device']
        self.hidden_dim = param['hidden_dim']
        self.input_dim = param['input_dim']
        self.num_layers = param['num_layers']
        self.padding_value = param['padding_value']
        self.max_seq_len = param['max_seq_len']
        self.module = param['module']
        if self.module.lower()=='lstm':
            self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True).to(self.device)
        elif self.module.lower()=='gru':
            self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True).to(self.device)
        else:
            raise ValueError('Only support LSTM and GRU')
        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
        self.act = nn.Sigmoid()

        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.fc.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
    def forward(self, x,T=None):
        x=x.to(self.device)
        # T=T.to(self.device)
        output, _ = self.rnn(x)
        linear = self.fc(output)
        supervised = self.act(linear)
        # Shape should be (batch_size, seq_len, hidden_dim)
        return supervised

class TimeGAN(nn.Module):
    def __init__(self, param):
        super(TimeGAN, self).__init__()
        self.device=param['device']
        self.hidden_dim = param['hidden_dim']
        self.input_dim = param['input_dim']
        self.num_layers = param['num_layers']
        self.padding_value = param['padding_value']
        self.max_seq_len = param['max_seq_len']
        self.batch_size = param['batch_size']
        self.module = param['module']
        self.Z_dim=param['Z_dim']
        self.embedder = Embedder(param)
        self.generator = Generator(param)
        self.discriminator = Discriminator(param)
        self.recovery = Recovery(param)
        self.supervisor = Supervisor(param)
    def _recovery_forward(self, X, T):
        """The embedding network forward pass and the embedder network loss
        Args:
            - X: the original input features
            - T: the temporal information
        Returns:
            - E_loss: the reconstruction loss
            - X_tilde: the reconstructed features
        """
        # Forward Pass
        ## X = (Batch, Seq_len, Features)
        ## T = [Seq_len, Seq_len, ...]
        X=X.to(self.device)
        T=T.to(self.device)

        H = self.embedder(X, T)
        H=H.to(self.device)
        ### H= (Batch, Seq_len, Hidden_dim)

        X_tilde = self.recovery(H, T)
        X_tilde=X_tilde.to(self.device)
        # X_tilde = (Batch, Seq_len, input_dim)

        # For Joint training
        H_hat_supervise = self.supervisor(H, T)
        H_hat_supervise=H_hat_supervise.to(self.device)
        G_loss_S = torch.nn.functional.mse_loss(
            H_hat_supervise[:,:-1], 
            H[:,1:]
        ) # Teacher forcing next output
        
        # Reconstruction Loss
        E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, X)
        E_loss0 = 10 * torch.sqrt(E_loss_T0)
        E_loss = E_loss0 + 0.1 * G_loss_S
        return E_loss, E_loss0, E_loss_T0
    def _recovery_Xtilde(self,X,T):
         # Forward Pass
        ## X = (Batch, Seq_len, Features)
        ## T = [Seq_len, Seq_len, ...]
        X=X.to(self.device)
        T=T.to(self.device)
        H = self.embedder(X, T)
        H=H.to(self.device)
        ### H= (Batch, Seq_len, Hidden_dim)
        X_tilde = self.recovery(H, T)
        X_tilde=X_tilde.to(self.device)
        return X_tilde
    def _Embeder_Xtilde(self,X,T):
         # Forward Pass
        ## X = (Batch, Seq_len, Features)
        ## T = [Seq_len, Seq_len, ...]
        X=X.to(self.device)
        T=T.to(self.device)
        H = self.embedder(X, T)
        H=H.to(self.device)
        ### H= (Batch, Seq_len, Hidden_dim)
        return H
    def _supervisor_forward(self, X, T):
        """The supervisor training forward pass
        Args:
            - X: the original feature input
        Returns:
            - S_loss: the supervisor's loss
        """
        # Supervision Forward Pass
         # Forward Pass
        ## X = (Batch, Seq_len, Features)
        ## T = [Seq_len, Seq_len, ...]
        X=X.to(self.device)
        T=T.to(self.device)
        H = self.embedder(X, T)
        H=H.to(self.device)
        ### H= (Batch, Seq_len, Hidden_dim)
        H_hat_supervise = self.supervisor(H, T)
        H_hat_supervise=H_hat_supervise.to(self.device)

        # Supervised loss
        S_loss = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1], H[:,1:])        # Teacher forcing next output
        return S_loss
    def _discriminator_forward(self, X, T, Z, gamma=1):
        """The discriminator forward pass and adversarial loss
        Args:
            - X: the input features
            - T: the temporal information
            - Z: the input noise
        Returns:
            - D_loss: the adversarial loss
        """
        # Real
        X=X.to(self.device)
        T=T.to(self.device)
        Z=Z.to(self.device)
        H = self.embedder(X, T).detach()
        H= H.to(self.device)
        # Generator
        # Z = (Batch, Seq_len, Z_dim)
        E_hat = self.generator(Z, T).detach()
        E_hat=E_hat.to(self.device)
        # E_hat = (Batch, Seq_len, hidden_dim)

        H_hat = self.supervisor(E_hat, T).detach()
        H_hat=H_hat.to(self.device)
        # Forward Pass
        Y_real = self.discriminator(H, T)            # Encoded original data
        Y_real=Y_real.to(self.device)
        Y_fake = self.discriminator(H_hat, T)        # Output of generator + supervisor
        Y_fake=Y_fake.to(self.device)
        Y_fake_e = self.discriminator(E_hat, T)      # Output of generator
        Y_fake_e=Y_fake_e.to(self.device)
        D_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(Y_real, torch.ones_like(Y_real))
        D_loss_real=D_loss_real.to(self.device)
        D_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.zeros_like(Y_fake))
        D_loss_fake=D_loss_fake.to(self.device)
        D_loss_fake_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.zeros_like(Y_fake_e))
        D_loss_fake_e=D_loss_fake_e.to(self.device)

        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        return D_loss
    def _generator_forward(self, X, T, Z, gamma=1):
        """The generator forward pass
        Args:
            - X: the original feature input
            - T: the temporal information
            - Z: the noise for generator input
        Returns:
            - G_loss: the generator's loss
        """
        # Supervisor Forward Pass
        # Forward Pass
        ## X = (Batch, Seq_len, Features)
        ## T = [Seq_len, Seq_len, ...]
        X=X.to(self.device)
        T=T.to(self.device)
        Z=Z.to(self.device)
        H = self.embedder(X, T)
        H=H.to(self.device)
        ### H= (Batch, Seq_len, Hidden_dim)
        H_hat_supervise = self.supervisor(H, T)
        H_hat_supervise=H_hat_supervise.to(self.device)

        # Generator Forward Pass
        # Z = (Batch, Seq_len, Z_dim)
        E_hat = self.generator(Z, T)
        E_hat=E_hat.to(self.device)
        # E_hat = (Batch, Seq_len, hidden_dim)
        H_hat = self.supervisor(E_hat, T)
        H_hat=H_hat.to(self.device)
        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)
        X_hat=X_hat.to(self.device)
        # Generator Loss
        # 1. Adversarial loss
        Y_fake = self.discriminator(H_hat, T)        # Output of supervisor
        Y_fake=Y_fake.to(self.device)
        Y_fake_e = self.discriminator(E_hat, T)      # Output of generator
        Y_fake_e=Y_fake_e.to(self.device)
        G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
        G_loss_U=G_loss_U.to(self.device)
        G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e))
        G_loss_U_e=G_loss_U_e.to(self.device)
        # 2. Supervised loss
        G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1], H[:,1:])        # Teacher forcing next output
        G_loss_S=G_loss_S.to(self.device)
        # 3. Two Momments
        G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
        G_loss_V1=G_loss_V1.to(self.device)
        G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))
        G_loss_V2=G_loss_V2.to(self.device)
        G_loss_V = G_loss_V1 + G_loss_V2
        G_loss_V=G_loss_V.to(self.device)
        # 4. Summation
        G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V
        G_loss=G_loss.to(self.device)
        return G_loss
    def _inference(self, Z, T):
        """Inference for generating synthetic data
        Args:
            - Z: the input noise
            - T: the temporal information
        Returns:
            - X_hat: the generated data
        """
        # Generator Forward Pass
        # Z = (Batch, Seq_len, Z_dim)
        Z=Z.to(self.device)
        # T=T.to(self.device)
        E_hat = self.generator(Z, T)
        E_hat=E_hat.to(self.device)
        # E_hat = (Batch, Seq_len, hidden_dim)
        H_hat = self.supervisor(E_hat, T)
        H_hat=H_hat.to(self.device)
        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)
        X_hat=X_hat.to(self.device)
        # Shape should be (batch_size, seq_len, input_dim)
        return X_hat
    def forward(self, X, T, Z, obj, gamma=1):
        """
        Args:
            - X: the input features (B, H, F)
            - T: the temporal information (B)
            - Z: the sampled noise (B, H, Z)
            - obj: the network to be trained (`autoencoder`, `supervisor`, `generator`, `discriminator`)
            - gamma: loss hyperparameter
        Returns:
            - loss: The loss for the forward pass
            - X_hat: The generated data
        """
        if obj != "inference":
            if X is None:
                raise ValueError("`X` should be given")

            # X = torch.FloatTensor(X)
            # X = X.to(self.device)
            X=torch.tensor(X, dtype=torch.float32, device=self.device)

        if Z is not None:
            # Z = torch.FloatTensor(Z)
            # Z = Z.to(self.device)
            Z=torch.tensor(Z, dtype=torch.float32, device=self.device)
        if obj == "autoencoder":
            # Embedder & Recovery
            loss = self._recovery_forward(X, T)

        elif obj == "supervisor":
            # Supervisor
            loss = self._supervisor_forward(X, T)

        elif obj == "generator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Generator
            loss = self._generator_forward(X, T, Z)

        elif obj == "discriminator":
            if Z is None:
                raise ValueError("`Z` is not given")
            
            # Discriminator
            loss = self._discriminator_forward(X, T, Z)
            
            return loss

        elif obj == "inference":

            X_hat = self._inference(Z, T)
            X_hat = X_hat.to(self.device)

            return X_hat
        
        elif obj=='autoencoder_out':
            return self._recovery_Xtilde(X,T)

        elif obj=='embeder_out':
            return self._Embeder_Xtilde(X,T)
        else: 
            raise ValueError("`obj` should be either `autoencoder`, `supervisor`, `generator`, or `discriminator`")
        return loss
