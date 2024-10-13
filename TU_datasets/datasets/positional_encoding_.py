import torch
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
import torch.nn.functional as F
import pdb

def compute_laplacian(edge_index, num_nodes, edge_weight=None):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
    
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device).scatter_add_(0, row, edge_weight)
    
    L = torch.diag(deg) - torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes]))
    return L

def add_laplacian_eigenvector_pe(edge_index, num_nodes, k=6, is_undirected=False):
    L = compute_laplacian(edge_index, num_nodes)
    L_dense = L.to_dense().numpy()  # Convert to dense numpy array for eigen computation
    
    # Compute the smallest eigenvalues and corresponding eigenvectors
    if is_undirected:
        # Symmetric matrix, can use eigsh for faster computation
        eig_vals, eig_vecs = eigsh(L_dense, k=k + 1, which='SM')
    else:
        # For non-symmetric cases, use eig (though this should typically still be symmetric)
        eig_vals, eig_vecs = np.linalg.eig(L_dense)
    
    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])  # Sort eigenvectors based on eigenvalues
    pe = torch.from_numpy(eig_vecs[:, 1:k + 1]).float()  # Skip the first trivial eigenvector
    
    sign = -1 + 2 * torch.randint(0, 2, (k, ))

    # Calculate how many zeros need to be added
    current_length = pe.shape[1]
    padding_length = k - current_length

    # Perform padding
    pe = F.pad(pe, (0, padding_length), 'constant', 0)
    pe *= sign
    
    return pe