import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CausalDataset(Dataset):
    """Base class for environment-aware causal datasets."""
    def __init__(self, observations, latents, environments):
        self.observations = torch.FloatTensor(observations)
        self.latents = torch.FloatTensor(latents)
        self.environments = torch.LongTensor(environments)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return {
            'x': self.observations[idx],
            'z': self.latents[idx],
            'e': self.environments[idx]
        }

def get_dataloader(dataset, batch_size=64, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
