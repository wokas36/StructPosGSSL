from GCL.augmentors.augmentor import Graph, Augmentor
import pdb

class Identity(Augmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, g: Graph) -> Graph:
        return g
