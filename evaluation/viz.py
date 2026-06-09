import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import torch

def plot_causal_graph(adj, labels=None, title="Recovered Causal Graph"):
    plt.figure(figsize=(8, 6))
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    if labels:
        mapping = {i: labels[i] for i in range(len(labels))}
        G = nx.relabel_nodes(G, mapping)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            edge_color='gray', node_size=2000, font_size=12, 
            arrowsize=20, connectionstyle='arc3,rad=0.1')
    
    plt.title(title)
    plt.show()

def plot_latents(z_est, e, title="Latent Space"):
    plt.figure(figsize=(10, 8))
    # If high dim, use PCA
    if z_est.shape[1] > 2:
        from sklearn.decomposition import PCA
        z_est = PCA(n_components=2).fit_transform(z_est)
        
    scatter = plt.scatter(z_est[:, 0], z_est[:, 1], c=e, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Environment')
    plt.title(title)
    plt.xlabel("Comp 1")
    plt.ylabel("Comp 2")
    plt.show()

def plot_loss_curves(metrics_history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['rec'], label='Reconstruction')
    plt.plot(metrics_history['kl'], label='VAE KL')
    plt.title("Reconstruction & KL Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['h'], label='DAG Constraint (h)')
    plt.yscale('log')
    plt.title("DAG Constraint Violation")
    plt.legend()
    plt.show()
