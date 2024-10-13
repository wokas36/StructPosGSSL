from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import drop_node


class NodeDropping(Augmentor):
    def __init__(self, pn: float):
        super(NodeDropping, self).__init__()
        self.pn = pn

    def augment(self, g: Graph) -> Graph:
        x, edge_index, y, pos, edge_attr, edge_attr_v2, batch, ptr = g.unfold()

        edge_index, edge_attr, edge_mask = drop_node(edge_index, edge_attr, keep_prob=1. - self.pn)
        
        edge_attr_v2 = edge_attr_v2[edge_mask]

        return Graph(x=x, edge_index=edge_index, y=y, pos=pos, edge_attr=edge_attr, edge_attr_v2=edge_attr_v2,
                    batch=batch, ptr=ptr)