"""
utils for processing data used for training and evaluation
"""
import itertools
from copy import deepcopy as c

import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch
from scipy import linalg
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import to_networkx, from_networkx
import torch_geometric.utils as pyg_utils
import torch_geometric.transforms as T
import pdb

import torch
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
import torch.nn.functional as F

from GCL.augmentors.positional_encoding import add_laplacian_eigenvector_pe, calculate_centrality_measures

def extract_edge_attributes(data, pos_enc_dim):
    
    networkx_G = to_networkx(data, to_undirected=True)
    data.edge_attr_v2 = calculate_centrality_measures(networkx_G, data.edge_index)
    
    if data.x is None:
        pe = add_laplacian_eigenvector_pe(data.edge_index, data.num_nodes, k=pos_enc_dim, is_undirected=True) # IMDBBINARY
    else:
        pe = add_laplacian_eigenvector_pe(data.edge_index, data.x.shape[0], k=pos_enc_dim, is_undirected=True) # MUTAG | PTC
        
    data.pos = torch.tensor(pe, dtype=torch.float32)
    
    if data.edge_attr is None:
        new_edge_attr = torch.zeros((data.edge_index.size(1),), dtype=torch.int64)
        data.edge_attr = F.one_hot(new_edge_attr, num_classes=4).to(torch.float)
    
    return data