from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import dropout_edge


class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, y, pos, edge_attr, edge_attr_v2, batch, ptr = g.unfold()
        edge_index, edge_mask = dropout_edge(edge_index, p=1. - self.pe)
        
        edge_attr = edge_attr[edge_mask]
        edge_attr_v2 = edge_attr_v2[edge_mask]
        
        return Graph(x=x, edge_index=edge_index, y=y, pos=pos, edge_attr=edge_attr, edge_attr_v2=edge_attr_v2,
                    batch=batch, ptr=ptr)