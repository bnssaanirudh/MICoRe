import torch
import torch.nn as nn
from .ivae import MLP

class StandardVAE(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden_dim=64):
        super().__init__()
        self.encoder = MLP(obs_dim, hidden_dim, latent_dim * 2)
        self.decoder = MLP(latent_dim, hidden_dim, obs_dim)

    def forward(self, x, e=None):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z
