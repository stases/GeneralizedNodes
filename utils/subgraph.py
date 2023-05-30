import torch
from .tools import catch_lone_sender, fully_connected_edge_index, fully_connect_existing_nodes
from torch_geometric.transforms import BaseTransform

class Subgraph:
    def __init__(self, graph, mode='fractal', fully_connect=False, depth=1):
        self.device = graph.x.device
        self.num_nodes = graph.x.shape[0]
        self.subgraph = graph.clone().to(self.device)
        self.mode = mode
        self.fully_connect = fully_connect
        #TODO: not have this hardcoded
        self.crop_onehot = None # 5 is specific for QM9
        if 'transformer' in mode:
            self.mode = 'transformer'
            # split a string by underscore, for example transformer_3 into transformer and 3
            self.transformer_size = int(mode.split('_')[1])

    def convert_to_subgraph(self):
        if self.fully_connect:
            self.add_fully_connected_edges()
        self.add_subnode_features()
        self.add_subnode_forces()
        self.add_subatom_index()
        self.add_subnode_position()
        self.add_node_flags()
        self.add_subnode_edges()
        self.add_node_subnode_edges()
        self.add_subnode_node_edges()
        self.add_subgraph_batch_index()

        return self.subgraph

    def add_fully_connected_edges(self):
        # This uses fully_connect_existing_nodes from tools.py, as we want to get a fully connecte graph in a given range
        # We don't want to connect accidentally nodes that are not in the subgraph, and vice-versa
        self.subgraph.edge_index = fully_connect_existing_nodes(edge_index=self.subgraph.edge_index).to(self.device)
        #self.subgraph.subgraph_edge_index = fully_connect_existing_nodes(edge_index=self.subgraph.subgraph_edge_index).to(self.device)

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

    def add_subnode_position(self):
        if self.mode == 'fractal':
            self.subgraph.pos = self.subgraph.pos.repeat(self.num_nodes+1,1)
        # In the case of the transformer, the position has no interpretable meaning so we don't se it.
        elif self.mode == 'transformer':
            # Fill up subgraph.pos with zero positions so it matches self.total_num_nodes
            self.subgraph.pos = torch.cat([self.subgraph.pos, torch.zeros(self.total_num_nodes-self.num_nodes, 3).to(self.device)], dim=0)
            # asser that subgraph pos[0] is equal to total num of nodes
            assert self.subgraph.pos.shape[0] == self.total_num_nodes


    def add_subnode_forces(self):
        if hasattr(self.subgraph, 'force'):
            if self.mode == 'fractal':
                self.subgraph.force = self.subgraph.force.repeat(self.num_nodes+1,1)
            elif self.mode == 'transformer':
                # do same as for positions
                self.subgraph.force = torch.cat([self.subgraph.force, torch.zeros(self.total_num_nodes-self.num_nodes, 3).to(self.device)], dim=0)
                assert self.subgraph.force.shape[0] == self.total_num_nodes

    def add_node_flags(self):
        if hasattr(self.subgraph, 'x'):
            self.subgraph.ground_node = torch.arange(self.subgraph.x.shape[0]) < self.num_nodes
        else:
            raise ValueError('No node features found. Please add node features first.')

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

    def add_subatom_index(self):
        if self.mode == 'fractal':
            self.subgraph.subatom_index = torch.arange(self.num_nodes).repeat(self.num_nodes+1)
