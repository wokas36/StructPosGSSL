"""
General GNN framework
"""
from copy import deepcopy as c
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, LayerNorm, InstanceNorm, PairNorm, GraphSizeNorm, global_add_pool
# from layers.gine import GINEConv
from layers.feature_encoder import FeatureConcatEncoder
from torch.nn import Sequential, Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout
from torch_sparse import SparseTensor, matmul, sum as sp_sum
from torch_geometric.utils import add_self_loops
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
import math

def clones(module, N):
    """Layer clone function, used for concise code writing
    Args:
        module (nn.Module): the layer want to clone
        N (int): the time of clone
    """
    return nn.ModuleList(c(module) for _ in range(N))

class GNN(nn.Module):
    """A generalized GNN framework
    Args:
        num_layer (int): the number of GNN layer
        gnn_layer (nn.Module): gnn layer used in GNN model
        init_emb (nn.Module): initial node feature encoding
        num_hop1_edge (int): number of edge type at 1 hop
        max_edge_count (int): maximum count per edge type for encoding
        max_hop_num (int): maximum number of hop to consider in peripheral node configuration
        max_distance_count (int): maximum count per hop for encoding
        JK (str):method of jumping knowledge, last,concat,max or sum
        norm_type (str): method of normalization, batch or layer
        virtual_node (bool): whether to add virtual node in the model
        residual (bool): whether to add residual connection
        use_rd (bool): whether to add resistance distance as additional feature
        wo_peripheral_edge (bool): If true, remove peripheral edge information from model
        wo_peripheral_configuration (bool): If true, remove peripheral node configuration from model
        drop_prob (float): dropout rate
    """

    def __init__(self, num_layer, gnn_layer, init_emb, init_edge_attr_emb, init_edge_attr_v2_emb, JK="last", 
                 norm_type="batch", virtual_node=True, residual=False, drop_prob=0.1):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = gnn_layer.output_size
        self.K = gnn_layer.K
        self.dropout = nn.Dropout(drop_prob)
        self.JK = JK
        self.residual = residual
        self.virtual_node = virtual_node
        if self.JK == "concat":
            self.output_proj = nn.Sequential(nn.Linear((self.num_layer + 1) * self.hidden_size, self.hidden_size),
                                             nn.ReLU(), nn.Dropout(drop_prob))
        else:
            self.output_proj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
                                             nn.Dropout(drop_prob))

        if self.JK == "attention":
            self.attention_lstm = nn.LSTM(self.hidden_size, self.num_layer, 1, batch_first=True, bidirectional=True,
                                          dropout=0.)


        # embedding start from 1
        self.init_proj = init_emb
        self.init_edge_attr_emb = init_edge_attr_emb
        self.atom_encoder = AtomEncoder(emb_dim = init_emb.init_proj.out_features)
        self.bond_encoder = BondEncoder(emb_dim = init_edge_attr_emb.init_proj.out_features)
        
        self.init_edge_attr_v2_emb = init_edge_attr_v2_emb
        if self.virtual_node:
            # set the initial virtual node embedding to 0.
            self.virtualnode_embedding = torch.nn.Embedding(1, self.hidden_size)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            # List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()
            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    torch.nn.BatchNorm1d(self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    torch.nn.BatchNorm1d(self.hidden_size),
                    torch.nn.ReLU()))

        # gnn layer list
        self.gnns = clones(gnn_layer, num_layer)
        # norm list
        if norm_type == "Batch":
            self.norms = clones(BatchNorm(self.hidden_size), num_layer)
        elif norm_type == "Layer":
            self.norms = clones(LayerNorm(self.hidden_size), num_layer)
        elif norm_type == "Instance":
            self.norms = clones(InstanceNorm(self.hidden_size), num_layer)
        elif norm_type == "GraphSize":
            self.norms = clones(GraphSizeNorm(), num_layer)
        elif norm_type == "Pair":
            self.norms = clones(PairNorm(), num_layer)
        else:
            raise ValueError("Not supported norm method")

        self.reset_parameters()

    def weights_init(self, m):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.init_proj.reset_parameters()
        for g in self.gnns:
            g.reset_parameters()
        if self.JK == "attention":
            self.attention_lstm.reset_parameters()

        self.output_proj.apply(self.weights_init)
        if self.virtual_node:
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            self.mlp_virtualnode_list.apply(self.weights_init)

    def forward(self, data):
        edge_index, edge_attr, edge_attr_v2, batch = data.edge_index, data.edge_attr, data.edge_attr_v2,  data.batch
        
        # initial projection
        x = self.init_proj(data).squeeze() 
        
        edge_attr_embedding = self.init_edge_attr_emb(data)
        edge_attr_v2_embedding = self.init_edge_attr_v2_emb(data)
        
        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
            )

        # forward in gnn layer
        h_list = [x]
        for l in range(self.num_layer):
            if self.virtual_node:
                h_list[l] = h_list[l] + virtualnode_embedding[batch]
            h = self.gnns[l](h_list[l], edge_index, edge_attr_embedding, edge_attr_v2_embedding, batch)
            h = self.norms[l](h)
            # if not the last gnn layer, add dropout layer
            
            ######
            '''if l == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.dropout, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training = self.training)'''
            ########
            
            if l != self.num_layer - 1:
                h = self.dropout(h)

            if self.residual:
                h = h + h_list[l]

            h_list.append(h)

            if self.virtual_node:
                # update the virtual nodes
                if l < self.num_layer - 1:
                    virtualnode_embedding_temp = global_add_pool(
                        h_list[l], batch
                    ) + virtualnode_embedding
                    # transform virtual nodes using MLP

                    if self.residual:
                        virtualnode_embedding = virtualnode_embedding + self.dropout(
                            self.mlp_virtualnode_list[l](virtualnode_embedding_temp))
                    else:
                        virtualnode_embedding = self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))
        
        # JK connection
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            node_representation = F.max_pool1d(torch.cat(h_list, dim=-1), kernel_size=self.num_layer + 1).squeeze()
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)
        elif self.JK == "attention":
            h_list = [h.unsqueeze(0) for h in h_list]
            h_list = torch.cat(h_list, dim=0).transpose(0, 1)  # N *num_layer * H
            self.attention_lstm.flatten_parameters()
            attention_score, _ = self.attention_lstm(h_list)  # N * num_layer * 2*num_layer
            attention_score = torch.softmax(torch.sum(attention_score, dim=-1), dim=1).unsqueeze(
                -1)  # N * num_layer  * 1
            node_representation = torch.sum(h_list * attention_score, dim=1)

        return self.output_proj(node_representation)