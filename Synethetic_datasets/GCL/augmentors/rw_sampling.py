import torch
import torch_geometric.utils as pyg_utils
import torch.nn.functional as F
import networkx as nx
from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import random_walk_subgraph
from .positional_encoding import add_laplacian_eigenvector_pe, calculate_centrality_measures
import pdb

class RWSampling(Augmentor):
    def __init__(self, num_seeds: int, walk_length: int, pos_enc_dim: int = 6):
        super(RWSampling, self).__init__()
        self.num_seeds = num_seeds
        self.walk_length = walk_length
        self.pos_enc_dim = pos_enc_dim

    def augment(self, g: Graph) -> Graph:
        x, edge_index, y, pos, edge_attr, edge_attr_v2, batch, ptr = g.unfold()

        edge_index, edge_attr, edge_mask = random_walk_subgraph(edge_index, edge_attr, batch_size=self.num_seeds, 
                                                        length=self.walk_length)
        
        edge_attr_v2 = edge_attr_v2[edge_mask]

        return Graph(x=x, edge_index=edge_index, y=y, pos=pos, edge_attr=edge_attr, edge_attr_v2=edge_attr_v2,
                    batch=batch, ptr=ptr)