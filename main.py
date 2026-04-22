import argparse
import torch
import numpy as np
import os
from data.synthetic_gen import SyntheticGenerator
from data.data_manager import get_dataloader
from models.micore_plus import MICoRePlus
from training.trainer import MICoReTrainer
from evaluation.metrics import compute_dci, compute_graph_metrics, compute_mcc
from evaluation.viz import plot_causal_graph, plot_latents, plot_loss_curves

def run_experiment(args):
    # 1. Setup Data
    print(f"Generating {args.dataset} dataset...")
    if args.dataset == 'pendulum':
        dataset = SyntheticGenerator.generate_pendulum(num_samples=args.samples)
        adj_gt = np.array([[0, 0], [1, 0]]) # z1 -> z2
        latent_dim = 2
        obs_dim = 4
    elif args.dataset == '3dident':
        dataset, adj_gt = SyntheticGenerator.generate_3dident_mock(
            num_samples=args.samples, latent_dim=args.latent_dim, obs_dim=args.obs_dim
        )
        latent_dim = args.latent_dim
        obs_dim = args.obs_dim
    else:
        raise ValueError("Unknown dataset")

    num_envs = len(np.unique(dataset.environments.numpy()))
    train_loader = get_dataloader(dataset, batch_size=args.batch_size)

    # 2. Setup Model
    config = {
        'lr': args.lr,
        'lambda_rec': 1.0,
        'lambda_kl': 1.0,
        'lambda_graph': 1.0,
        'lambda_mi_kl': args.lambda_mi,
        'lambda_mi_delta': args.lambda_mi * 0.1,
        'lambda_adj_l1': 0.1,
        'dag_alpha': 0.0,
        'dag_rho': 1.0
    }

    model = MICoRePlus(latent_dim, obs_dim, num_envs, hidden_dim=args.hidden_dim)
    trainer = MICoReTrainer(model, train_loader, config, device=args.device)

    # 3. Train
    trainer.train(num_epochs=args.epochs)

    # 4. Evaluate
    print("\nEvaluating MICoRe...")
    model.eval()
    with torch.no_grad():
        x = dataset.observations.to(args.device)
        e = dataset.environments.to(args.device)
        outputs = model(x, e)
        
        z_est = outputs['mu'].cpu().numpy()
        z_gt = dataset.latents.numpy()
        adj_est = outputs['adj'].cpu().numpy()
        
        dci = compute_dci(z_est, z_gt)
        mcc = compute_mcc(z_est, z_gt)
        graph_metrics = compute_graph_metrics(adj_est, adj_gt)
        
        print("-" * 30)
        print("Identifiability & Disentanglement Metrics:")
        print(f"  MCC (Mean Correlation Coefficient): {mcc:.4f}")
        for k, v in dci.items():
            print(f"  {k}: {v:.4f}")
        print("\nCausal Graph Metrics:")
        for k, v in graph_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("-" * 30)

    # 5. Visualizations
    if args.viz:
        plot_causal_graph(adj_est, title=f"Recovered Graph ({args.dataset})")
        plot_latents(z_est, dataset.environments.numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='3dident', choices=['pendulum', '3dident'])
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--latent_dim', type=int, default=6)
    parser.add_argument('--obs_dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lambda_mi', type=float, default=1.0, help='Minimal Intervention weight')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--viz', action='store_true', help='Show plots')
    
    args = parser.parse_args()
    run_experiment(args)
