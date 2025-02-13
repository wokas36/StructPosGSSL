from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import permute


class NodeShuffling(Augmentor):
    def __init__(self):
        super(NodeShuffling, self).__init__()

    def augment(self, g: Graph) -> Graph:
        x, edge_index, y, pos, edge_attr, edge_attr_v2, batch, ptr = g.unfold()
        x, pos = permute(x, pos)
        return Graph(x=x, edge_index=edge_index, y=y, pos=pos, edge_attr=edge_attr, edge_attr_v2=edge_attr_v2, 
                     batch=batch, ptr=ptr)