import torch
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
import torch.nn.functional as F
import networkx as nx
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
        eig_vals, eig_vecs = eigsh(L_dense, k=k + 1, which='SM', ncv=50) #, ncv=100 -> COLLAB
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

def calculate_edge_clustering_coefficient(G):
    ecc = {}
    for u, v in G.edges():
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        common_neighbors = neighbors_u.intersection(neighbors_v)
        union_neighbors = (neighbors_u.union(neighbors_v)) - {u, v}
        if len(union_neighbors) == 0:
            ecc[(u, v)] = 0  # Avoid division by zero; no potential triangles if no union neighbors
        else:
            ecc[(u, v)] = len(common_neighbors) / len(union_neighbors)
    return ecc

def calculate_edge_closeness_centrality(G):
    edge_closeness = {}
    for u, v in G.edges():
        shortest_paths_u = nx.single_source_shortest_path_length(G, u)
        shortest_paths_v = nx.single_source_shortest_path_length(G, v)
        sum_distances = sum(shortest_paths_u.values()) + sum(shortest_paths_v.values()) - shortest_paths_u[v] - shortest_paths_v[u]
        edge_closeness[(u, v)] = (len(G.nodes) - 1) / sum_distances if sum_distances > 0 else 0
    return edge_closeness

def calculate_centrality_measures(networkx_G, edge_index):
    # Calculate edge betweenness centrality for graphs
    ebc = nx.edge_betweenness_centrality(networkx_G)
    
    # Calculate edge clustering coefficients for graphs
    ec = calculate_edge_clustering_coefficient(networkx_G)
    
    # Calculate edge closeness centrality for graphs
    ecc = calculate_edge_closeness_centrality(networkx_G)
    
    # Convert to tensors and consider both directions
    edge_attr_dict = {}
    for i, (u, v) in enumerate(edge_index.t().tolist()):
        # Ensure both directions are considered
        if (u, v) in ebc:
            edge_attr_dict[(u, v)] = [ebc[(u, v)], ec[(u, v)], ecc[(u, v)]]
        elif (v, u) in ebc:
            edge_attr_dict[(u, v)] = [ebc[(v, u)], ec[(v, u)], ecc[(v, u)]]
            
    # Form the final edge_attr tensor
    edge_attrs_v2 = [edge_attr_dict[(u, v)] for u, v in edge_index.t().tolist()]
    edge_attr_v2 = torch.tensor(edge_attrs_v2)
    
    return edge_attr_v2