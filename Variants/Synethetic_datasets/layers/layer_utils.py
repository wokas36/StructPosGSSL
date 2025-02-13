"""
Utils for defining model layers
"""
from layers.KHopGNNConv import *
import pdb

def make_gnn_layer(args):
    """function to construct gnn layer
    Args:
        args (argparser): arguments list
    """
    model_name = args.model_name
    if model_name == "KHopGNNConv":
        gnn_layer = KHopGNNConv(args.hidden_size, args.hidden_size, args.hidden_size, args.hidden_size, 
                                args.K, args.drop_prob, args.pos_attr, args.combine)
    else:
        raise ValueError("Not supported GNN type")

    return gnn_layer