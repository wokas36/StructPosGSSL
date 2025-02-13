import torch
import torch_geometric.utils as pyg_utils
import torch.nn.functional as F
import networkx as nx
from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import compute_ppr
from .positional_encoding import add_laplacian_eigenvector_pe, calculate_centrality_measures

class PPRDiffusion(Augmentor):
    def __init__(self, alpha: float = 0.2, eps: float = 1e-4, use_cache: bool = True, add_self_loop: bool = True, 
                 pos_enc_dim: int = 6):
        super(PPRDiffusion, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop
        self.pos_enc_dim = pos_enc_dim

    def augment(self, g: Graph) -> Graph:
        if self._cache is not None and self.use_cache:
            return self._cache
        
        x, edge_index, y, pos, edge_attr, edge_attr_v2, batch, ptr = g.unfold()
        edge_index, edge_attr, edge_attr_v2 = compute_ppr(
            edge_index, edge_attr, edge_attr_v2, 
            alpha=self.alpha, eps=self.eps, ignore_edge_attr=False, add_self_loop=self.add_self_loop
        )
        
        res = Graph(x=x, edge_index=edge_index, y=y, pos=pos, edge_attr=edge_attr, edge_attr_v2=edge_attr_v2,
                    batch=batch, ptr=ptr)
        self._cache = res
        return res
