import torch
from .subgraph import Subgraph
from .tools import catch_lone_sender, fully_connected_edge_index
from torch_geometric.transforms import BaseTransform

class Fully_Connected_Graph(BaseTransform):
    def __init__(self):
        pass
    def __call__(self, data):
        num_nodes = data.x.shape[0]
        edge_index = fully_connected_edge_index(num_nodes)
        data.edge_index = edge_index
        return data

class To_OneHot(BaseTransform):
    def __init__(self, num_classes=None):
        self.num_classes = num_classes
    def __call__(self, data):
        if self.num_classes is None:
            self.num_classes = data.x.max().item() + 1
        data.x = torch.nn.functional.one_hot(data.x, self.num_classes).squeeze()
        return data

class Rename_MD17_Features(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.x = data.z
        data.z = None
        return data

class Graph_to_Subgraph(BaseTransform):
    def __init__(self, mode='fractal', depth=1, fully_connect=False):
        self.mode = mode
        self.depth = depth
        self.fully_connect = fully_connect
    def __call__(self, data):
        subgraph = Subgraph(data, mode=self.mode, depth=self.depth, fully_connect=self.fully_connect)
        return subgraph.convert_to_subgraph()