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
        data.edge_index = edge_index.to(data.edge_index.device)
        return data

class Graph_to_Subgraph(BaseTransform):
    def __init__(self, mode='fractal', depth=1, fully_connect=False):
        self.mode = mode
        self.depth = depth
        self.fully_connect = fully_connect
    def __call__(self, data):
        subgraph = Subgraph(data, mode=self.mode, depth=self.depth, fully_connect=self.fully_connect)
        return subgraph.convert_to_subgraph()