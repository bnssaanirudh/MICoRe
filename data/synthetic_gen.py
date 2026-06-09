import numpy as np
import torch
import torch.nn as nn
from .data_manager import CausalDataset

class SyntheticGenerator:
    """Generates synthetic datasets for causal representation learning."""
    
    @staticmethod
    def generate_pendulum(num_samples=5000, num_envs=3):
        """
        Simulates two coupled pendulums.
        z1 -> z2
        """
        all_x = []
        all_z = []
        all_e = []
        
        samples_per_env = num_samples // num_envs
        
        for e in range(num_envs):
            # Intervene on noise or mean
            mean_z1 = 0.5 * e
            z1 = np.random.normal(mean_z1, 0.1, samples_per_env)
            
            # Causal link z1 -> z2
            z2 = 0.7 * z1 + np.random.normal(0.2 * e, 0.05, samples_per_env)
            
            z = np.stack([z1, z2], axis=1)
            
            # Non-linear mapping to high-dim space
            # x = [sin(z1), cos(z1), sin(z2), cos(z2)] + noise
            x = np.stack([
                np.sin(z1), np.cos(z1),
                np.sin(z2), np.cos(z2)
            ], axis=1)
            x += np.random.normal(0, 0.01, x.shape)
            
            all_x.append(x)
            all_z.append(z)
            all_e.append(np.full(samples_per_env, e))
            
        return CausalDataset(
            np.concatenate(all_x),
            np.concatenate(all_z),
            np.concatenate(all_e)
        )

    @staticmethod
    def generate_3dident_mock(num_samples=10000, num_envs=5, latent_dim=10, obs_dim=64):
        """
        Mock Causal3DIdent with complex causal structure.
        Uses a random MLP as the 'renderer'.
        """
        # Fixed random 'renderer'
        renderer = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, obs_dim),
            nn.Sigmoid()
        )
        for param in renderer.parameters():
            param.requires_grad = False

        all_x = []
        all_z = []
        all_e = []
        
        samples_per_env = num_samples // num_envs
        
        # Ground truth DAG (sparse)
        # Random DAG for latent variables
        adj = np.zeros((latent_dim, latent_dim))
        for i in range(latent_dim):
            for j in range(i + 1, latent_dim):
                if np.random.rand() < 0.3:
                    adj[i, j] = 1.0 # i -> j
        
        for e in range(num_envs):
            z = np.zeros((samples_per_env, latent_dim))
            # Generate latents according to DAG
            for i in range(latent_dim):
                # Parents
                parents = np.where(adj[:, i] == 1)[0]
                if len(parents) > 0:
                    val = np.dot(z[:, parents], np.random.uniform(0.5, 1.5, len(parents)))
                else:
                    val = 0
                
                # Soft intervention: change noise distribution in environment e
                # Only intervene on a few variables per environment
                if (i + e) % 3 == 0:
                    noise_mean = 0.5 * e
                    noise_std = 0.2
                else:
                    noise_mean = 0.0
                    noise_std = 0.1
                    
                z[:, i] = val + np.random.normal(noise_mean, noise_std, samples_per_env)
            
            with torch.no_grad():
                z_torch = torch.FloatTensor(z)
                x = renderer(z_torch).numpy()
            
            all_x.append(x)
            all_z.append(z)
            all_e.append(np.full(samples_per_env, e))
            
        return CausalDataset(
            np.concatenate(all_x),
            np.concatenate(all_z),
            np.concatenate(all_e)
        ), adj
