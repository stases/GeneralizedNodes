import torch
from torch_geometric.transforms import BaseTransform

class Subgraph:
    def __init__(self, graph, mode='fractal', depth=1):
        self.device = graph.x.device
        self.num_nodes = graph.x.shape[0]
        self.subgraph = graph.clone().to(self.device)
        self.mode = mode
        #TODO: not have this hardcoded
        self.crop_onehot = 5
        if 'transformer' in mode:
            self.mode = 'transformer'
            # split a string by underscore, for example transformer_3 into transformer and 3
            self.transformer_size = int(mode.split('_')[1])

    def convert_to_subgraph(self):
        self.add_subnode_features()
        self.add_node_flags()
        self.add_subnode_edges()
        self.add_node_subnode_edges()
        self.add_subnode_node_edges()
        self.add_subgraph_batch_index()
        return self.subgraph

    def add_subnode_features(self):
        if self.crop_onehot:
            # take only the self.crop_onehot first features
            self.subgraph.x = self.subgraph.x[:, :self.crop_onehot]
        if self.mode == 'fractal':
            self.subgraph.x = self.subgraph.x.repeat(self.num_nodes+1,1)
        self.num_features = self.subgraph.x.shape[1]
        self.total_num_nodes = self.subgraph.x.shape[0]

        if self.mode == 'transformer':
            self.subgraph.x = torch.cat([self.subgraph.x, torch.zeros(self.num_nodes, self.transformer_size).to(self.device)], dim=1)
            # create nodes for the transformer, where each node is one hot encoded to the transformer_size
            transformer_nodes = torch.eye(self.transformer_size)
            # now append zeros of the transformer_size at dimension 1
            transformer_nodes = torch.cat([torch.zeros(self.transformer_size, self.num_features), transformer_nodes], dim=1).to(self.device)
            # add transformer nodes to the subgraph for every node
            self.subgraph.x = torch.cat([self.subgraph.x, transformer_nodes.repeat(self.num_nodes,1)], dim=0)
        # Total number of nodes after adding the subgraph structure
        self.total_num_nodes = self.subgraph.x.shape[0]

    def add_node_flags(self):
        if hasattr(self.subgraph, 'x'):
            self.subgraph.ground_node = torch.arange(self.subgraph.x.shape[0]) < self.num_nodes
        else:
            print('No node features found. Please add node features first.')

    def add_subnode_edges(self):
        if self.mode == 'fractal':
            self.subgraph.subgraph_edge_index = self.subgraph.edge_index + self.num_nodes
            for subg in range(self.num_nodes-1):
                #TODO: Check if this is correct
                self.subgraph.subgraph_edge_index = torch.cat([self.subgraph.subgraph_edge_index,
                                                               self.subgraph.edge_index + self.num_nodes + (subg+1)*self.num_nodes],
                                                              dim=1)
        elif self.mode == 'transformer':
            # create a fully connected edge index for the transformer of size transformer_size
            transformer_edge_index = torch.stack([torch.arange(self.transformer_size).repeat_interleave(self.transformer_size),
                                                  torch.arange(self.transformer_size).repeat(self.transformer_size)],
                                                 dim=0).to(self.device)
            self.subgraph.subgraph_edge_index = transformer_edge_index + self.num_nodes
            for subg in range(self.num_nodes-1):
                #TODO: Check if this is correct
                self.subgraph.subgraph_edge_index = torch.cat([self.subgraph.subgraph_edge_index,
                                                               transformer_edge_index  + self.num_nodes + (subg+1)*self.transformer_size],
                                                              dim=1).to(self.device)

    def add_node_subnode_edges(self):
        if self.mode == 'fractal':
            self.subgraph.node_subnode_index = torch.stack([torch.arange(self.num_nodes).repeat_interleave(self.num_nodes),
                                                            torch.arange(self.num_nodes, self.total_num_nodes)],
                                                           dim=0).to(self.device)
        elif self.mode == 'transformer':
            self.subgraph.node_subnode_index = torch.stack([torch.arange(self.num_nodes).repeat_interleave(self.transformer_size),
                                                            torch.arange(self.num_nodes, self.total_num_nodes)],
                                                           dim=0).to(self.device)

    def add_subnode_node_edges(self):
        if self.mode == 'fractal':
            self.subgraph.subnode_node_index = torch.stack([torch.arange(self.num_nodes, self.total_num_nodes), torch.arange(self.num_nodes).repeat_interleave(self.num_nodes)], dim=0)
        elif self.mode == 'transformer':
            self.subgraph.subnode_node_index = torch.stack([torch.arange(self.num_nodes, self.total_num_nodes),
                                                            torch.arange(self.num_nodes).repeat_interleave(self.transformer_size)],
                                                           dim=0).to(self.device)

    def add_subgraph_batch_index(self):
        if self.mode == 'fractal':
            self.subgraph.subgraph_batch_index = torch.arange(self.num_nodes).repeat_interleave(self.num_nodes)
        elif self.mode == 'transformer':
            self.subgraph.subgraph_batch_index = torch.arange(self.num_nodes).repeat_interleave(self.transformer_size)

class Graph_to_Subgraph(BaseTransform):
    def __init__(self, mode='fractal', depth=1):
        self.mode = mode
        self.depth = depth
    def __call__(self, data):
        subgraph = Subgraph(data, mode=self.mode, depth=self.depth)
        return subgraph.convert_to_subgraph()

