import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from scipy.optimize import linear_sum_assignment
import torch

def compute_dci(z_est, z_gt):
    """
    Computes Disentanglement, Completeness, and Informativeness.
    z_est: [N, latent_dim] - estimated latents
    z_gt: [N, true_latent_dim] - ground truth latents
    """
    N, D_est = z_est.shape
    _, D_gt = z_gt.shape
    
    # Informativeness: Error in predicting z_gt from z_est
    # Disentanglement & Completeness: Based on Importance Matrix R
    # R_ij: importance of z_est_i in predicting z_gt_j
    
    importance_matrix = np.zeros((D_est, D_gt))
    
    for j in range(D_gt):
        model = RandomForestRegressor(n_estimators=10)
        model.fit(z_est, z_gt[:, j])
        # Use feature importance as proxy for R
        importance_matrix[:, j] = model.feature_importances_
        
    # Disentanglement
    # D_i = 1 - entropy(R_i)
    # Normalized R_i
    R_norm_row = importance_matrix / (importance_matrix.sum(axis=1, keepdims=True) + 1e-9)
    entropy_row = -np.sum(R_norm_row * np.log(R_norm_row + 1e-9), axis=1) / np.log(D_gt + 1e-9)
    disentanglement = np.mean(1 - entropy_row)
    
    # Completeness
    # C_j = 1 - entropy(R_j)
    R_norm_col = importance_matrix / (importance_matrix.sum(axis=0, keepdims=True) + 1e-9)
    entropy_col = -np.sum(R_norm_col * np.log(R_norm_col + 1e-9), axis=0) / np.log(D_est + 1e-9)
    completeness = np.mean(1 - entropy_col)
    
    # Informativeness
    # Prediction error
    z_pred = np.zeros_like(z_gt)
    for j in range(D_gt):
        model = RandomForestRegressor(n_estimators=10)
        model.fit(z_est, z_gt[:, j])
        z_pred[:, j] = model.predict(z_est)
    informativeness = -mean_squared_error(z_gt, z_pred)
    
    return {
        'disentanglement': disentanglement,
        'completeness': completeness,
        'informativeness': informativeness
    }

def compute_mcc(z_est, z_gt):
    """
    Computes the Mean Correlation Coefficient (MCC) using Hungarian matching.
    This is the canonical metric for identifiability in iVAEs.
    """
    N, D_est = z_est.shape
    _, D_gt = z_gt.shape
    
    # We need D_est == D_gt for a perfect 1-to-1 matching, 
    # but Hungarian works for rectangular cost matrices too.
    
    # Compute correlation matrix between est and gt
    corr_matrix = np.zeros((D_est, D_gt))
    for i in range(D_est):
        for j in range(D_gt):
            # Pearson correlation
            # Handle constant arrays which give NaN correlation
            if np.std(z_est[:, i]) == 0 or np.std(z_gt[:, j]) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(z_est[:, i], z_gt[:, j])[0, 1]
            corr_matrix[i, j] = np.abs(corr)
            
    # We want to maximize correlation, so we minimize the negative correlation
    cost_matrix = -corr_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    mcc = corr_matrix[row_ind, col_ind].mean()
    return mcc

def compute_graph_metrics(adj_est, adj_gt, threshold=0.1):
    """
    Structural Hamming Distance, Precision, Recall.
    """
    est = (np.abs(adj_est) > threshold).astype(int)
    gt = (np.abs(adj_gt) > threshold).astype(int)
    
    shd = np.sum(np.abs(est - gt))
    
    tp = np.sum((est == 1) & (gt == 1))
    fp = np.sum((est == 1) & (gt == 0))
    fn = np.sum((est == 0) & (gt == 1))
    
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    
    return {
        'shd': shd,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
