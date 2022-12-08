import torch
from torch_geometric.transforms import BaseTransform

class Subgraph:
    def __init__(self, graph, depth=1):
        self.num_nodes = graph.x.shape[0]
        self.subgraph = graph.clone()

    def convert_to_subgraph(self):
        self.add_subnode_features()
        self.add_node_flags()
        self.add_subnode_edges()
        self.add_node_subnode_edges()
        self.add_subnode_node_edges()
        self.add_subgraph_batch_index()
        return self.subgraph

    def add_subnode_features(self):
        self.subgraph.x = self.subgraph.x.repeat(self.num_nodes+1,1)
        self.total_num_nodes = self.subgraph.x.shape[0]

    def add_node_flags(self):
        if hasattr(self.subgraph, 'x'):
            self.subgraph.ground_node = torch.arange(self.subgraph.x.shape[0]) < self.num_nodes
        else:
            print('No node features found. Please add node features first.')

    def add_subnode_edges(self):
        self.subgraph.subgraph_edge_index = self.subgraph.edge_index + self.num_nodes
        for subg in range(self.num_nodes):
            self.subgraph.subgraph_edge_index = torch.cat([self.subgraph.subgraph_edge_index, self.subgraph.edge_index + (subg+1)*self.num_nodes], dim=1)

    def add_node_subnode_edges(self):
        self.subgraph.node_subnode_index = torch.stack([torch.arange(self.num_nodes).repeat_interleave(self.num_nodes), torch.arange(self.num_nodes, self.total_num_nodes)], dim=0)

    def add_subnode_node_edges(self):
        self.subgraph.subnode_node_index = torch.stack([torch.arange(self.num_nodes, self.total_num_nodes), torch.arange(self.num_nodes).repeat_interleave(self.num_nodes)], dim=0)

    def add_subgraph_batch_index(self):
        self.subgraph.subgraph_batch_index = torch.arange(self.num_nodes).repeat_interleave(self.num_nodes)

class Graph_to_Subgraph(BaseTransform):
    def __call__(self, data):
        subgraph = Subgraph(data)
        return subgraph.convert_to_subgraph()