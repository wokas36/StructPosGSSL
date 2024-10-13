from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List


class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    y: torch.LongTensor
    pos: torch.FloatTensor
    edge_attr: Optional[torch.FloatTensor]
    edge_attr_v2:Optional[torch.FloatTensor]
    batch:Optional[torch.LongTensor]
    ptr:Optional[torch.LongTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor, 
                              Optional[torch.FloatTensor], Optional[torch.FloatTensor], Optional[torch.LongTensor],
                              Optional[torch.LongTensor]]:
        return self.x, self.edge_index, self.y, self.pos, self.edge_attr, self.edge_attr_v2, self.batch, self.ptr


class Augmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(self, x: torch.FloatTensor, edge_index: torch.LongTensor, y: torch.LongTensor, pos: torch.FloatTensor,
                 edge_attr: Optional[torch.FloatTensor], edge_attr_v2: Optional[torch.FloatTensor], 
                 batch:Optional[torch.LongTensor], ptr:Optional[torch.LongTensor]
    ) -> Graph: return self.augment(Graph(x, edge_index, y, pos, edge_attr, edge_attr_v2, batch, ptr))


class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: Graph) -> Graph:
        for aug in self.augmentors:
            g = aug.augment(g)
        return g


class RandomChoice(Augmentor):
    def __init__(self, augmentors: List[Augmentor], num_choices: int):
        super(RandomChoice, self).__init__()
        assert num_choices <= len(augmentors)
        self.augmentors = augmentors
        self.num_choices = num_choices

    def augment(self, g: Graph) -> Graph:
        num_augmentors = len(self.augmentors)
        perm = torch.randperm(num_augmentors)
        idx = perm[:self.num_choices]
        for i in idx:
            aug = self.augmentors[i]
            g = aug.augment(g)
        return g