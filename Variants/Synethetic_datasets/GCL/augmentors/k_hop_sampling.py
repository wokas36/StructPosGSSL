from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import k_hop_subgraph
import pdb

class KHopSampling(Augmentor):
    def __init__(self, hops: int, sample_size: int):
        super(KHopSampling, self).__init__()
        self.hops = hops
        self.sample_size = sample_size

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        pdb.set_trace()

        edge_index, edge_weights = k_hop_subgraph(edge_index, edge_weights, hops=self.hops, sample_size=self.sample_size)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)