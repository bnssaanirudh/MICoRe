import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from .losses import micore_loss

class MICoReTrainer:
    def __init__(self, model, train_loader, config, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        # NOTEARS parameters
        self.dag_alpha = 0.0
        self.dag_rho = 1.0
        self.h_val = float('inf')

    def train_epoch(self):
        self.model.train()
        total_metrics = {}
        
        for batch in self.train_loader:
            x = batch['x'].to(self.device)
            e = batch['e'].to(self.device)
            z_gt = batch['z'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Update config with current AL parameters
            self.config['dag_alpha'] = self.dag_alpha
            self.config['dag_rho'] = self.dag_rho
            
            outputs = self.model(x, e)
            outputs['model'] = self.model # Pass model for MI loss calculation
            
            loss_dict = micore_loss(outputs, x, e, self.config)
            loss_dict['total'].backward()
            
            self.optimizer.step()
            
            # Accumulate metrics
            for k, v in loss_dict.items():
                if k != 'total':
                    total_metrics[k] = total_metrics.get(k, 0) + v.item()
                    
        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in total_metrics.items()}

    def train(self, num_epochs=100):
        print(f"Starting training for {num_epochs} epochs...")
        h_prev = float('inf')
        
        for epoch in range(num_epochs):
            metrics = self.train_epoch()
            h_curr = metrics['h']
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Rec={metrics['rec']:.4f}, KL={metrics['kl']:.4f}, H={h_curr:.6f}, MI_KL={metrics['mi_kl']:.4f}")
            
            # Augmented Lagrangian update (every 10 epochs or so)
            if epoch > 0 and epoch % 20 == 0:
                if h_curr > 0.25 * h_prev:
                    self.dag_rho *= 10
                else:
                    self.dag_alpha += self.dag_rho * h_curr
                h_prev = h_curr
                
                # Check for DAG convergence
                if h_curr < 1e-8:
                    print("DAG constraint satisfied.")
                    # We can stop early or continue for representation learning
                    
        return metrics
