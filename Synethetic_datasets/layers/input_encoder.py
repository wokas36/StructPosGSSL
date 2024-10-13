"""
different input encoder for different dataset
"""

import torch
import torch.nn as nn
import pdb

# class EmbeddingEncoder(nn.Module):
#     """Encoder with embedding layer
#     Args:
#         input_size (int): input size of feature
#         hidden_size (int): hidden size of encoder
#     """
#     def __init__(self, input_size, hidden_size):
#         super(EmbeddingEncoder, self).__init__()
#         self.init_proj = nn.Embedding(input_size, hidden_size)

#     def reset_parameters(self):
#         self.init_proj.reset_parameters()

#     def forward(self, data):
#         return self.init_proj(data.x)


class LinearEncoder(nn.Module):
    """Encoder with linear projection layer
    Args:
        input_size (int): input size of feature
        hidden_size (int): hidden size of encoder
    """
    def __init__(self, input_size, hidden_size, pos_attr=False):
        super(LinearEncoder, self).__init__()
        self.init_proj = nn.Linear(input_size, hidden_size)
        self.pos_attr = pos_attr

    def reset_parameters(self):
        self.init_proj.reset_parameters()

    def forward(self, data):
        # return self.init_proj(data.x) if not self.pos_attr else self.init_proj(data.pos)
        return self.init_proj(data.x.to(torch.float)) if not self.pos_attr else self.init_proj(data.pos.to(torch.float))
    
    
class LinearEdgeEncoder(nn.Module):
    """Encoder with linear projection layer
    Args:
        input_size (int): input size of position
        hidden_size (int): hidden size of encoder
    """
    def __init__(self, input_size, hidden_size, edge_attr=True):
        super(LinearEdgeEncoder, self).__init__()
        self.init_proj = nn.Linear(input_size, hidden_size)
        self.edge_attr = edge_attr

    def reset_parameters(self):
        self.init_proj.reset_parameters()

    def forward(self, data):
        # return self.init_proj(data.edge_attr) if self.edge_attr else self.init_proj(data.edge_attr_v2)
        return self.init_proj(data.edge_attr.to(torch.float)) if self.edge_attr else self.init_proj(data.edge_attr_v2.to(torch.float))


class QM9InputEncoder(nn.Module):
    """Input encoder for QM9 dataset
    Args:
        hidden_size (int): hidden size of encoder
        use_pos (bool): If true, add 3D position information
    """
    def __init__(self, hidden_size, use_pos=False):
        super(QM9InputEncoder, self).__init__()
        self.use_pos = use_pos
        if use_pos:
            input_size = 22
        else:
            input_size = 19
        self.init_proj = nn.Linear(input_size, hidden_size)
        self.z_embedding = nn.Embedding(1000, 8)

    def reset_parameters(self):
        self.init_proj.reset_parameters()
        self.z_embedding.reset_parameters()

    def forward(self, data):
        x = data.x
        z = data.z
        if "pos" in data:
            pos = data.pos
        else:
            pos = None

        z_emb = 0
        if z is not None:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)

        # concatenate with continuous node features
        x = torch.cat([z_emb, x], -1)

        if self.use_pos:
            x = torch.cat([x, pos], 1)

        x = self.init_proj(x)
        return x
