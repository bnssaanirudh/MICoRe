import torch
import torch.nn as nn
from .ivae import iVAE
from .notears import NOTEARS_MLP

class MICoRePlus(nn.Module):
    def __init__(self, latent_dim, obs_dim, num_envs, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.num_envs = num_envs
        
        # iVAE base
        self.ivae = iVAE(latent_dim, obs_dim, num_envs, hidden_dim)
        
        # NOTEARS module on latents
        self.notears = NOTEARS_MLP(latent_dim, hidden_dim=hidden_dim//2)
        
    def forward(self, x, u):
        # 1. iVAE pass
        x_recon, mu, logvar, z, p_mu, p_logvar = self.ivae(x, u)
        
        # 2. NOTEARS pass (graph learning on z)
        # We model Z = f(Z) + epsilon, so the exogenous noise is epsilon = Z - f(Z)
        z_pred = self.notears(mu) 
        epsilon_mu = mu - z_pred
        
        return {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'p_mu': p_mu,
            'p_logvar': p_logvar,
            'z_pred': z_pred,
            'epsilon_mu': epsilon_mu,
            'adj': self.notears.get_adj()
        }

    def get_intervention_loss(self, u, num_samples=100):
        """
        Computes the Minimal Intervention Loss L_MI.
        Instead of a parametric L1 shift on mu/var, this uses the 1D Wasserstein distance 
        on samples drawn from the prior, allowing for non-parametric interventions.
        L_MI = lambda * sum_i W_1(P_u, P_0)_i
        """
        # P_obs is environment 0
        obs_idx = torch.zeros_like(u)
        mu_0 = self.ivae.prior_mu(obs_idx)
        logvar_0 = self.ivae.prior_logvar(obs_idx)
        
        mu_u = self.ivae.prior_mu(u)
        logvar_u = self.ivae.prior_logvar(u)
        
        # 1. Draw samples from the priors: shape (batch_size, num_samples, latent_dim)
        std_0 = torch.exp(0.5 * logvar_0).unsqueeze(1)
        std_u = torch.exp(0.5 * logvar_u).unsqueeze(1)
        
        eps_0 = torch.randn(u.size(0), num_samples, self.latent_dim, device=u.device)
        eps_u = torch.randn(u.size(0), num_samples, self.latent_dim, device=u.device)
        
        samples_0 = mu_0.unsqueeze(1) + eps_0 * std_0
        samples_u = mu_u.unsqueeze(1) + eps_u * std_u
        
        # 2. Compute 1D Wasserstein distance per dimension
        # Sort samples along the num_samples dimension
        sorted_0, _ = torch.sort(samples_0, dim=1)
        sorted_u, _ = torch.sort(samples_u, dim=1)
        
        # W_1 distance per dimension i: mean(|sorted_u - sorted_0|) over samples
        w1_dist = torch.mean(torch.abs(sorted_u - sorted_0), dim=1)
        
        # Intervention mask Δ_u is the Wasserstein distance per latent dimension
        delta_u = w1_dist
        
        # Also return KL between shifted and base prior as a secondary metric or regularizer if needed
        var_u = torch.exp(logvar_u)
        var_0 = torch.exp(logvar_0)
        kl = 0.5 * (logvar_0 - logvar_u + (var_u + (mu_u - mu_0)**2) / var_0 - 1.0)
        kl = kl.sum(dim=1) 
        
        return kl.mean(), delta_u.mean()
