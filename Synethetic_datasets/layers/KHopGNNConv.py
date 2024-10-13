import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_dense_adj
from torch_sparse import SparseTensor, matmul, sum as sp_sum
import torch_geometric.utils as pyg_utils
from copy import deepcopy as c
import math

from .combine import *

class KHopGNNConv(MessagePassing):
    """
    Args:
        input_size (int): the size of input feature
        output_size (int): the size of output feature
        K (int): number of hop to consider in Convolution layer
        num_hop1_edge (int): number of edge type at 1 hop
        num_pe (int): maximum number of path encoding, larger or equal to 1
        combine (str): combination method for information in different hop. select from(geometric, attention)
    """

    def __init__(self, input_size, output_size, edge_attr_size, edge_attr_v2_size, K, dropout, pos_attr, combine="geometric"):
        super(KHopGNNConv, self).__init__(node_dim=0)
        self.aggr = "add"
        self.K = K
        self.input_size = input_size
        self.output_size = output_size
        self.edge_attr_size = edge_attr_size
        self.edge_attr_v2_size = edge_attr_v2_size
        self.pos_attr = pos_attr

        self.mlp = torch.nn.Sequential(torch.nn.Linear(output_size, output_size), torch.nn.BatchNorm1d(output_size), 
                                       torch.nn.ReLU(), torch.nn.Linear(output_size, output_size))
        
        self.eps = torch.nn.Parameter(torch.FloatTensor(1))
        # self.atom_encoder = AtomEncoder(emb_dim = output_size)
        # self.bond_encoder = BondEncoder(emb_dim = output_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv_eps = 0.1 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)

    def forward(self, x, edge_index, edge_attr_embedding, edge_attr_v2_embedding, batch):

        batch_num_node = x.size(0)
        
        adj = pyg_utils.to_dense_adj(edge_index).squeeze()
        _, value = pyg_utils.dense_to_sparse(adj)
        
        current_adj = adj
        mask = torch.eye(adj.shape[0])
        
        agg_out = self.propagate(edge_index=edge_index, x=x, norm=value, edge_attr=edge_attr_embedding,
                           edge_attr_v2=edge_attr_v2_embedding)
        
        if not self.pos_attr:
            for i in range(1, self.K):
                current_adj = current_adj@adj
                current_diagonal = torch.diagonal(current_adj, 0)
                current_mask = (1. - mask)*current_adj
                current_off_diagonal = current_mask/current_mask.sum(1, keepdim=True)
                current_off_diagonal[torch.isnan(current_off_diagonal)] = 0
                new_adj = mask*torch.diag_embed(current_diagonal) + current_off_diagonal
                edge_index_k, value_k = pyg_utils.dense_to_sparse(new_adj)

                out = self.propagate(edge_index=edge_index_k, x=x, norm=value_k)
                agg_out += out
            
        agg_out = self.mlp((1+self.eps) * x + agg_out)

        return agg_out
    
    def message(self, x_j, norm, edge_attr=None, edge_attr_v2=None):
        if edge_attr==None and edge_attr_v2==None:
            x_j = norm.view(-1, 1)*F.relu(x_j)
        else:
            x_j = norm.view(-1, 1)*F.relu(x_j + edge_attr + edge_attr_v2)
        return x_j

    def update(self, aggr_out):
        return aggr_out