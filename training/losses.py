import torch
import torch.nn.functional as F
from models.notears import dag_constraint

def compute_vae_kl(mu, logvar, p_mu, p_logvar):
    """
    KL(q(z|x,e) || p(z|e))
    """
    var = torch.exp(logvar)
    p_var = torch.exp(p_logvar)
    kl = 0.5 * (p_logvar - logvar + (var + (mu - p_mu)**2) / p_var - 1.0)
    return kl.sum(dim=1).mean()

def micore_loss(outputs, targets, e, config):
    """
    Full MICoRe Loss.
    """
    # 1. Reconstruction Loss
    l_rec = F.mse_loss(outputs['x_recon'], targets)
    
    # 2. VAE KL (Identifiability on Exogenous Noise)
    # The prior is evaluated on epsilon = Z - f(Z; W)
    l_kl = compute_vae_kl(outputs['epsilon_mu'], outputs['logvar'], outputs['p_mu'], outputs['p_logvar'])
    
    # 3. NOTEARS Graph Loss (Z reconstruction)
    # The NOTEARS loss is usually ||Z - f(Z)||^2
    l_graph = F.mse_loss(outputs['z_pred'], outputs['mu'].detach()) 
    
    # 4. DAG Constraint
    adj = outputs['adj']
    h_val = dag_constraint(adj)
    
    # 5. Minimal Intervention Loss
    l_kl_mi, l_delta_mi = outputs['model'].get_intervention_loss(e)
    
    # 6. Sparsity on Adj
    l_adj_l1 = torch.norm(adj, p=1)
    
    total_loss = (
        config['lambda_rec'] * l_rec +
        config['lambda_kl'] * l_kl +
        config['lambda_graph'] * l_graph +
        config['lambda_mi_kl'] * l_kl_mi +
        config['lambda_mi_delta'] * l_delta_mi +
        config['lambda_adj_l1'] * l_adj_l1
    )
    
    # Augmented Lagrangian terms for DAG constraint
    # L = F(A) + alpha * h(A) + (rho/2) * h(A)^2
    l_dag = config['dag_alpha'] * h_val + 0.5 * config['dag_rho'] * (h_val ** 2)
    total_loss += l_dag
    
    return {
        'total': total_loss,
        'rec': l_rec,
        'kl': l_kl,
        'graph': l_graph,
        'mi_kl': l_kl_mi,
        'mi_delta': l_delta_mi,
        'h': h_val,
        'adj_l1': l_adj_l1
    }
