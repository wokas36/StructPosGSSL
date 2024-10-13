from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import drop_feature


class EdgeAttrMasking(Augmentor):
    def __init__(self, pf: float):
        super(EdgeAttrMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, y, pos, edge_attr, edge_attr_v2, batch, ptr = g.unfold()
        if edge_attr is not None:
            edge_attr = drop_feature(edge_attr, self.pf)
        if edge_attr_v2 is not None:
            edge_attr_v2 = drop_feature(edge_attr_v2, self.pf)
        return Graph(x=x, edge_index=edge_index, y=y, pos=pos, edge_attr=edge_attr, edge_attr_v2=edge_attr_v2,
                    batch=batch, ptr=ptr)
