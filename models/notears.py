import torch
import torch.nn as nn
import numpy as np

class LocallyConnected(nn.Module):
    """
    A layer where each output channel is a separate linear layer of the input.
    Used for NOTEARS-MLP to enforce the adjacency matrix.
    """
    def __init__(self, n_vars, input_dim, output_dim, bias=True):
        super().__init__()
        self.n_vars = n_vars
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # weights: [n_vars, input_dim, output_dim]
        # For variable i, we have a linear layer from all n_vars inputs to output_dim
        self.weight = nn.Parameter(torch.Tensor(n_vars, n_vars, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_vars, output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: [batch, n_vars]
        # We want output: [batch, n_vars, output_dim]
        # For each variable i: y_i = x * W_i + b_i
        # Weight [n_vars, n_vars, output_dim]
        # x expanded: [batch, n_vars, 1]
        x = x.unsqueeze(2) # [batch, n_vars, 1]
        
        # Batch matrix multiplication
        # [n_vars, batch, 1] @ [n_vars, 1, output_dim] -> [n_vars, batch, output_dim]
        # But x is [batch, n_vars]
        
        # Let's use einsum for clarity
        # x: [b, j] where j is input index
        # w: [i, j, k] where i is output variable, j is input variable, k is hidden dim
        # out: [b, i, k]
        out = torch.einsum('bj,ijk->bik', x.squeeze(2), self.weight)
        
        if self.bias is not None:
            out += self.bias
        return out

class NOTEARS_MLP(nn.Module):
    def __init__(self, n_vars, hidden_dim=16):
        super().__init__()
        self.n_vars = n_vars
        self.fc1 = LocallyConnected(n_vars, n_vars, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Each var maps back to 1 value
        
    def forward(self, z):
        # z: [batch, n_vars]
        h = torch.sigmoid(self.fc1(z)) # [batch, n_vars, hidden_dim]
        # Reshape for fc2: [batch * n_vars, hidden_dim]
        h = h.view(-1, h.size(-1))
        out = self.fc2(h)
        out = out.view(-1, self.n_vars) # [batch, n_vars]
        return out

    def get_adj(self):
        """
        Computes the weighted adjacency matrix from the first layer weights.
        A_ij = ||W_{i,j}||_2 (edge from j to i)
        """
        # weight: [n_vars, n_vars, hidden_dim]
        # norm over hidden_dim
        adj = torch.norm(self.fc1.weight, dim=2) # [n_vars, n_vars]
        # Diagonal should be 0 (no self-loops)
        adj = adj * (1 - torch.eye(self.n_vars, device=adj.device))
        return adj

def dag_constraint(adj):
    """
    h(A) = tr(exp(A o A)) - d
    """
    d = adj.shape[0]
    # Matrix exponential can be computed via power series or eigenvalue decomp
    # For small d, we can use torch.matrix_exp
    # A o A (Hadamard product)
    m = adj * adj
    # tr(exp(m))
    h = torch.trace(torch.matrix_exp(m)) - d
    return h
