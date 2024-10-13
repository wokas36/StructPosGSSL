from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import drop_feature


class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, y, pos, edge_attr, edge_attr_v2, batch, ptr = g.unfold()
        x = drop_feature(x, self.pf)
        pos = drop_feature(pos, self.pf)
        return Graph(x=x, edge_index=edge_index, y=y, pos=pos, edge_attr=edge_attr, edge_attr_v2=edge_attr_v2,
                    batch=batch, ptr=ptr)