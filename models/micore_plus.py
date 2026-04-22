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

    def get_intervention_loss(self, u):
        """
        Computes the Minimal Intervention Loss L_MI.
        L_MI = lambda * || theta_u - theta_0 ||_1
        Penalizes the shift in causal mechanisms (prior parameters) from the observational environment.
        """
        # P_obs is environment 0
        obs_idx = torch.zeros_like(u)
        mu_0 = self.ivae.prior_mu(obs_idx)
        logvar_0 = self.ivae.prior_logvar(obs_idx)
        
        mu_u = self.ivae.prior_mu(u)
        logvar_u = self.ivae.prior_logvar(u)
        
        # Intervention mask Δ_u: L1 norm of shift in causal mechanisms
        delta_mu = torch.abs(mu_u - mu_0)
        delta_var = torch.abs(logvar_u - logvar_0)
        delta_u = delta_mu + delta_var
        
        # Also return KL between shifted and base prior as a secondary metric or regularizer if needed
        var_u = torch.exp(logvar_u)
        var_0 = torch.exp(logvar_0)
        kl = 0.5 * (logvar_0 - logvar_u + (var_u + (mu_u - mu_0)**2) / var_0 - 1.0)
        kl = kl.sum(dim=1) 
        
        return kl.mean(), delta_u.mean()
