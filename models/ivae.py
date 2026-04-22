import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LeakyReLU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class iVAE(nn.Module):
    def __init__(self, latent_dim, obs_dim, num_envs, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.num_envs = num_envs
        
        # Encoder: q(z | x, e)
        # Note: iVAE identifiability doesn't strictly need e in encoder, 
        # but it helps with convergence.
        self.encoder = MLP(obs_dim + num_envs, hidden_dim, latent_dim * 2)
        
        # Decoder: p(x | z)
        self.decoder = MLP(latent_dim, hidden_dim, obs_dim)
        
        # Prior parameters: p(z | e) = N(mu_e, sigma_e^2)
        # We learn these as embeddings
        self.prior_mu = nn.Embedding(num_envs, latent_dim)
        self.prior_logvar = nn.Embedding(num_envs, latent_dim)
        
        # Initialize prior_mu to 0 and prior_logvar to 0 (std=1)
        nn.init.zeros_(self.prior_mu.weight)
        nn.init.zeros_(self.prior_logvar.weight)

    def encode(self, x, u):
        # One-hot encode u (the auxiliary environment variable)
        u_onehot = F.one_hot(u, self.num_envs).float()
        combined = torch.cat([x, u_onehot], dim=1)
        h = self.encoder(combined)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, u):
        mu, logvar = self.encode(x, u)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        
        # Get prior params p(epsilon|u)
        p_mu = self.prior_mu(u)
        p_logvar = self.prior_logvar(u)
        
        return x_recon, mu, logvar, z, p_mu, p_logvar
