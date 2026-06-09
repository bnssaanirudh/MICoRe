import asyncio
import os
import sys
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_gen import SyntheticGenerator
from data.data_manager import get_dataloader
from models.micore_plus import MICoRePlus
from training.trainer import MICoReTrainer
from evaluation.metrics import compute_dci, compute_graph_metrics, compute_mcc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to keep track of training
class TrainingState:
    def __init__(self):
        self.is_training = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.history = []
        self.final_results = None
        self.adj = None
        self.z_est = None
        self.env_labels = None

state = TrainingState()

class TrainConfig(BaseModel):
    dataset: str = '3dident'
    epochs: int = 50
    samples: int = 5000
    latent_dim: int = 6
    obs_dim: int = 32
    batch_size: int = 64
    lr: float = 1e-3
    lambda_mi: float = 1.0
    hidden_dim: int = 64

def run_training(config: TrainConfig):
    state.is_training = True
    state.current_epoch = 0
    state.total_epochs = config.epochs
    state.history = []
    state.final_results = None
    state.adj = None
    state.z_est = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config.dataset == 'pendulum':
        dataset = SyntheticGenerator.generate_pendulum(num_samples=config.samples)
        adj_gt = np.array([[0, 0], [1, 0]]) # z1 -> z2
        latent_dim = 2
        obs_dim = 4
    elif config.dataset == '3dident':
        dataset, adj_gt = SyntheticGenerator.generate_3dident_mock(
            num_samples=config.samples, latent_dim=config.latent_dim, obs_dim=config.obs_dim
        )
        latent_dim = config.latent_dim
        obs_dim = config.obs_dim
    else:
        state.is_training = False
        return

    num_envs = len(np.unique(dataset.environments.numpy()))
    train_loader = get_dataloader(dataset, batch_size=config.batch_size)

    trainer_config = {
        'lr': config.lr,
        'lambda_rec': 1.0,
        'lambda_kl': 1.0,
        'lambda_graph': 1.0,
        'lambda_mi_kl': config.lambda_mi,
        'lambda_mi_delta': config.lambda_mi * 0.1,
        'lambda_adj_l1': 0.1,
        'dag_alpha': 0.0,
        'dag_rho': 1.0
    }

    model = MICoRePlus(latent_dim, obs_dim, num_envs, hidden_dim=config.hidden_dim).to(device)
    trainer = MICoReTrainer(model, train_loader, trainer_config, device=device)

    h_prev = float('inf')

    for epoch in range(config.epochs):
        metrics = trainer.train_epoch()
        h_curr = metrics['h']
        
        state.current_epoch = epoch + 1
        
        # Augmented Lagrangian update
        if epoch > 0 and epoch % 20 == 0:
            if h_curr > 0.25 * h_prev:
                trainer.dag_rho *= 10
            else:
                trainer.dag_alpha += trainer.dag_rho * h_curr
            h_prev = h_curr

        # Compute mid-training MCC roughly every 5 epochs or end
        mcc = 0.0
        shd = 0.0
        if epoch % 5 == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.no_grad():
                x = dataset.observations.to(device)
                u = dataset.environments.to(device)
                outputs = model(x, u)
                z_est = outputs['mu'].cpu().numpy()
                z_gt = dataset.latents.numpy()
                adj_est = outputs['adj'].cpu().numpy()
                
                mcc = compute_mcc(z_est, z_gt)
                graph_metrics = compute_graph_metrics(adj_est, adj_gt)
                shd = graph_metrics['shd']
            model.train()

        state.history.append({
            'epoch': epoch + 1,
            'loss': metrics['total'],
            'rec': metrics['rec'],
            'kl': metrics['kl'],
            'h': metrics['h'],
            'mcc': mcc if mcc > 0 else (state.history[-1]['mcc'] if state.history else 0),
            'shd': shd if shd > 0 else (state.history[-1]['shd'] if state.history else 0)
        })

    model.eval()
    with torch.no_grad():
        x = dataset.observations.to(device)
        u = dataset.environments.to(device)
        outputs = model(x, u)
        
        z_est = outputs['mu'].cpu().numpy()
        z_gt = dataset.latents.numpy()
        adj_est = outputs['adj'].cpu().numpy()
        
        dci = compute_dci(z_est, z_gt)
        mcc = compute_mcc(z_est, z_gt)
        graph_metrics = compute_graph_metrics(adj_est, adj_gt)
        
        state.final_results = {
            'dci': dci,
            'mcc': mcc,
            'graph': graph_metrics
        }
        state.adj = adj_est.tolist()
        # Subsample latents for visualization
        indices = np.random.choice(len(z_est), min(1000, len(z_est)), replace=False)
        state.z_est = z_est[indices].tolist()
        state.env_labels = dataset.environments.numpy()[indices].tolist()

    state.is_training = False

@app.post("/api/train")
def start_training(config: TrainConfig, background_tasks: BackgroundTasks):
    if state.is_training:
        return {"status": "Already training"}
    background_tasks.add_task(run_training, config)
    return {"status": "Training started"}

@app.get("/api/status")
def get_status():
    return {
        "is_training": state.is_training,
        "current_epoch": state.current_epoch,
        "total_epochs": state.total_epochs,
        "history": state.history
    }

@app.get("/api/results")
def get_results():
    if state.is_training:
        return {"status": "Still training"}
    if state.final_results is None:
        return {"status": "No results available"}
    
    return {
        "metrics": state.final_results,
        "adj": state.adj,
        "z_est": state.z_est,
        "env_labels": state.env_labels
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
