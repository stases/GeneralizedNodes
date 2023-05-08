import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as geom_nn

from utils.tools import catch_lone_sender, fully_connected_edge_index
from ..layers.layers import FractalMP, MP, EGNNLayer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class TransformerNet(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, out_features, depth=1, pool="add", num_heads=1,
                 add_residual_skip=False):
        super().__init__()
        self.name = 'TransformerNet'
        self.depth = depth
        self.pool = pool
        self.add_residual_skip = add_residual_skip
        self.num_heads = num_heads
        self.embedding = nn.Linear(node_features, hidden_features)
        self.ground_mps = nn.ModuleList()
        self.ground_to_sub_mps = nn.ModuleList()
        self.sub_mps = nn.ModuleList()
        self.sub_to_ground_mps = nn.ModuleList()
        for i in range(depth):
            self.ground_mps.append(geom_nn.GATv2Conv(hidden_features, hidden_features, heads=num_heads))
            self.ground_to_sub_mps.append(geom_nn.GATv2Conv(hidden_features, hidden_features, heads=num_heads))
            self.sub_mps.append(geom_nn.GATv2Conv(hidden_features, hidden_features, heads=num_heads))
            self.sub_to_ground_mps.append(geom_nn.GATv2Conv(hidden_features, hidden_features, heads=num_heads))
        self.output = nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index, subgraph_edge_index, node_subnode_index, subnode_node_index, ground_node,
                subgraph_batch_index, batch_idx, edge_attr=None):
        x = self.embedding(x)

        for i in range(self.depth):
            if self.add_residual_skip:
                x_0 = x
            x = self.ground_mps[i](x, edge_index, edge_attr)
            # TODO: Check the order of edge indices; directed in which direction? subnode to node or vice versa
            x = self.ground_to_sub_mps[i](x, node_subnode_index, edge_attr)
            x = self.sub_mps[i](x, subgraph_edge_index, edge_attr)
            x = self.sub_to_ground_mps[i](x, subnode_node_index, edge_attr)
            if self.add_residual_skip:
                x = x + x_0
        # global pooling over nodes whose ground node is true
        if self.pool == "mean":
            x = tg.nn.global_mean_pool(x[ground_node], batch_idx)
        elif self.pool == "add":
            x = tg.nn.global_add_pool(x[ground_node], batch_idx)
        elif self.pool == "max":
            x = tg.nn.global_max_pool(x[ground_node], batch_idx)
        x = self.output(x)
        return x


class FractalNet(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, out_features, depth=1, pool="add",
                 add_residual_skip=False, masking=False, layernorm=False, pool_all=False,**kwargs):
        super().__init__()
        self.name = 'FractalNet'
        self.depth = depth
        self.pool = pool
        self.add_residual_skip = add_residual_skip
        self.masking = masking
        self.layernorm = layernorm
        self.pool_all = pool_all
        self.embedding = nn.Linear(node_features, hidden_features)
        self.ground_mps = nn.ModuleList()
        self.ground_to_sub_mps = nn.ModuleList()
        self.sub_mps = nn.ModuleList()
        self.sub_to_ground_mps = nn.ModuleList()
        if self.layernorm:
            self.ln = nn.ModuleList()
        for i in range(depth):
            self.ground_mps.append(MP(hidden_features, edge_features, hidden_features, hidden_features))
            self.ground_to_sub_mps.append(MP(hidden_features, edge_features, hidden_features, hidden_features))
            self.sub_mps.append(MP(hidden_features, edge_features, hidden_features, hidden_features))
            self.sub_to_ground_mps.append(MP(hidden_features, edge_features, hidden_features, hidden_features))
            if self.layernorm:
                self.ln.append(nn.LayerNorm(hidden_features))
        self.output_1 = nn.Linear(hidden_features, hidden_features)
        self.output_2 = nn.Linear(hidden_features, out_features)


    def forward(self, x, edge_index, subgraph_edge_index, node_subnode_index, subnode_node_index, ground_node,
                subgraph_batch_index, batch_idx, edge_attr=None):
        num_nodes = x.shape[0]
        x = self.embedding(x)

        # TODO: Is graph.y doing something weird with rescaling and normalizing etc? Shapes and stuff, or messing up the statistics
        for i in range(self.depth):
            if self.add_residual_skip:
                x_0 = x

            update_mask = catch_lone_sender(edge_index, num_nodes)
            x_backup = x[~update_mask]
            x = self.ground_mps[i](x, edge_index, edge_attr)
            if self.masking:
                x[~update_mask] = x_backup
            # TODO: Check the order of edge indices; directed in which direction? subnode to node or vice versa

            update_mask = catch_lone_sender(node_subnode_index, num_nodes)
            x_backup = x[~update_mask]
            x = self.ground_to_sub_mps[i](x, node_subnode_index, edge_attr)
            if self.masking:
                x[~update_mask] = x_backup

            update_mask = catch_lone_sender(subgraph_edge_index, num_nodes)
            x_backup = x[~update_mask]
            x = self.sub_mps[i](x, subgraph_edge_index, edge_attr)
            if self.masking:
                x[~update_mask] = x_backup

            update_mask = catch_lone_sender(subnode_node_index, num_nodes)
            x_backup = x[~update_mask]
            x = self.sub_to_ground_mps[i](x, subnode_node_index, edge_attr)
            if self.masking:
                x[~update_mask] = x_backup

            if self.layernorm:
                x = self.ln[i](x)

            if self.add_residual_skip:
                x = x + x_0

        # global pooling over nodes whose ground node is true
        if self.pool == "mean":
            if self.pool_all:
                x = tg.nn.global_mean_pool(x, batch_idx)
            else:
                x = tg.nn.global_mean_pool(x[ground_node], batch_idx)
        elif self.pool == "add":
            if self.pool_all:
                x = tg.nn.global_add_pool(x, batch_idx)
            else:
                x = tg.nn.global_add_pool(x[ground_node], batch_idx)
        elif self.pool == "max":
            if self.pool_all:
                x = tg.nn.global_max_pool(x, batch_idx)
            else:
                x = tg.nn.global_max_pool(x[ground_node], batch_idx)

        x = self.output_1(x)
        x = self.output_2(x)
        return x


class Net(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, out_features, depth=1, pool="add", add_residual_skip=False, layernorm=False, **kwargs):
        super().__init__()
        self.name = 'Net'
        self.depth = depth
        self.pool = pool
        self.layernorm = layernorm
        self.add_residual_skip = add_residual_skip
        self.embedding = nn.Linear(node_features, hidden_features)
        self.mps = nn.ModuleList()
        if self.layernorm:
            self.ln = nn.ModuleList()
        for i in range(depth):
            self.mps.append(MP(hidden_features, edge_features, hidden_features, hidden_features))
            if self.layernorm:
                self.ln.append(nn.LayerNorm(hidden_features))
        self.output_1 = nn.Linear(hidden_features, hidden_features)
        self.output_2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index, batch_idx, edge_attr=None):
        x = self.embedding(x)
        for i in range(self.depth):
            x_0 = x
            x = self.mps[i](x, edge_index, edge_attr)
            if self.layernorm:
                x = self.ln[i](x)
            if self.add_residual_skip:
                x = x + x_0

        if self.pool == "mean":
            x = tg.nn.global_mean_pool(x, batch_idx)
        elif self.pool == "add":
            x = tg.nn.global_add_pool(x, batch_idx)
        elif self.pool == "max":
            x = tg.nn.global_max_pool(x, batch_idx)
        x = self.output_1(x)
        x = self.output_2(x)
        return x

class EGNN(nn.Module):
    def __init__(self, node_features, hidden_features, out_features, depth, **kwargs):
        super().__init__()
        self.name = "EGNN"
        self.embed = nn.Sequential(nn.Linear(node_features, hidden_features), nn.SiLU(), nn.Linear(hidden_features, hidden_features))
        self.layers = nn.ModuleList([EGNNLayer(hidden_features) for _ in range(depth)])
        self.pre_readout = nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.SiLU(), nn.Linear(hidden_features, hidden_features))
        self.readout = nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.SiLU(), nn.Linear(hidden_features, out_features))

    def forward(self, data):
        x, pos, edge_index, batch_idx = data.x, data.pos, data.edge_index, data.batch
        x = self.embed(x)

        for layer in self.layers:
            x = x + layer(x, pos, edge_index)

        x = self.pre_readout(x)
        x = tg.nn.global_add_pool(x, batch_idx)
        out = self.readout(x)
        return out

class Fractal_EGNN(nn.Module):
    def __init__(self, node_features, hidden_features, out_features, depth, **kwargs):
        super().__init__()
        self.name = "Fractal_EGNN"
        self.embed = nn.Sequential(nn.Linear(node_features, hidden_features), nn.SiLU(), nn.Linear(hidden_features, hidden_features))
        self.ground_mps = nn.ModuleList([EGNNLayer(hidden_features) for _ in range(depth)])
        self.ground_to_sub_mps = nn.ModuleList([EGNNLayer(hidden_features) for _ in range(depth)])
        self.sub_mps = nn.ModuleList([EGNNLayer(hidden_features) for _ in range(depth)])
        self.sub_to_ground_mps = nn.ModuleList([EGNNLayer(hidden_features) for _ in range(depth)])
        self.pre_readout = nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.SiLU(), nn.Linear(hidden_features, hidden_features))
        self.readout = nn.Sequential(nn.Linear(hidden_features, hidden_features), nn.SiLU(), nn.Linear(hidden_features, out_features))

    def forward(self, data):
        x, pos, edge_index, batch_idx = data.x, data.pos, data.edge_index, data.batch
        x = self.embed(x)

        for layer in self.layers:
            x = x + layer(x, pos, edge_index)

        x = self.pre_readout(x)
        x = tg.nn.global_add_pool(x, batch_idx)
        out = self.readout(x)
        return out

class GNN(nn.Module):
    """implements a graphical neural network in pytorch. In particular, we will use pytorch geometric's nn_conv module so we can apply a neural network to the edges.
    """

    def __init__(
            self,
            n_node_features: int,
            n_edge_features: int,
            n_hidden: int,
            n_output: int,
            num_convolution_blocks: int,
    ) -> None:
        """create the gnn
        Args:
            n_node_features: input features on each node
            n_edge_features: input features on each edge
            n_hidden: hidden features within the neural architectures (embeddings, nodes after graph convolutions, etc.)
            n_output: how many output features
            num_convolution_blocks: how many blocks convolutions should be performed. A block may include multiple convolutions

        TODO:
        - define a GNN which has the following structure: node embedding -> [ReLU -> RGCNConv -> ReLU -> MFConv] x num_convs -> Add-Pool -> Linear -> ReLU -> Linear
        - One the data has been pooled, it may be beneficial to apply another MLP on the pooled data before predicing the output.
        """
        super().__init__()
        self.name = 'GNN'
        layers = []
        in_channels, out_channels = n_node_features, n_hidden

        self.embed = nn.Linear(n_node_features, n_hidden)
        for l_idx in range(num_convolution_blocks):
            layers += [
                nn.Identity() if l_idx == 0 else nn.ReLU(),
                geom_nn.RGCNConv(in_channels=n_hidden,
                                 out_channels=n_hidden,
                                 num_relations=n_edge_features),
                nn.ReLU(),
                geom_nn.MFConv(in_channels=n_hidden,
                               out_channels=n_hidden,
                               max_degree=10)
            ]
        self.GNN = nn.ModuleList(layers)
        self.linear_1 = nn.Linear(out_channels, out_channels)
        self.linear_2 = nn.Linear(out_channels, 1)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            edge_attr: edge attributes (pytorch geometric notation)
            batch_idx: Index of batch element for each node
        Returns:
            prediction
        TODO: implement the forward pass being careful to apply MLPs only where they are allowed!
        Hint: remember to use global pooling.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.embed(x)
        for layer in self.GNN:
            if (isinstance(layer, geom_nn.RGCNConv)):
                x = layer(x, edge_index, edge_attr)

            elif (isinstance(layer, geom_nn.MessagePassing)):
                x = layer(x, edge_index)

            else:
                x = layer(x)
        x = geom_nn.global_add_pool(x, batch_idx)
        x = F.relu(self.linear_1(x))
        out = self.linear_2(x)
        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device

class GNN_no_rel(nn.Module):
    """implements a graphical neural network in pytorch. In particular, we will use pytorch geometric's nn_conv module so we can apply a neural network to the edges.
    """

    def __init__(
            self,
            n_node_features: int,
            n_edge_features: int,
            n_hidden: int,
            n_output: int,
            num_convolution_blocks: int,
            pooling: str,
            **kwargs
    ) -> None:
        """create the gnn

        Args:
            n_node_features: input features on each node
            n_edge_features: input features on each edge
            n_hidden: hidden features within the neural architectures (embeddings, nodes after graph convolutions, etc.)
            n_output: how many output features
            num_convolution_blocks: how many blocks convolutions should be performed. A block may include multiple convolutions

        TODO:
        - define a GNN which has the following structure: node embedding -> [ReLU -> RGCNConv -> ReLU -> MFConv] x num_convs -> Add-Pool -> Linear -> ReLU -> Linear
        - One the data has been pooled, it may be beneficial to apply another MLP on the pooled data before predicing the output.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()
        self.name = 'GNN_no_rel'
        layers = []
        in_channels, out_channels = n_node_features, n_hidden

        self.embed = nn.Linear(n_node_features, n_hidden)
        for l_idx in range(num_convolution_blocks):
            layers += [
                nn.Identity() if l_idx == 0 else nn.ReLU(),
                geom_nn.MFConv(in_channels=n_hidden,
                               out_channels=n_hidden,
                               max_degree=10)
            ]
        self.GNN = nn.ModuleList(layers)
        self.linear_1 = nn.Linear(out_channels, out_channels)
        self.linear_2 = nn.Linear(out_channels, 1)
        self.pooling = pooling
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            edge_attr: edge attributes (pytorch geometric notation)
            batch_idx: Index of batch element for each node

        Returns:
            prediction

        TODO: implement the forward pass being careful to apply MLPs only where they are allowed!

        Hint: remember to use global pooling.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.embed(x)
        edge_index = edge_index.type(torch.int64)
        for layer in self.GNN:
            if (isinstance(layer, geom_nn.RGCNConv)):
                x = layer(x, edge_index, edge_attr)

            elif (isinstance(layer, geom_nn.MessagePassing)):
                x = layer(x, edge_index)

            else:
                x = layer(x)
        if self.pooling == 'add':
            x = geom_nn.global_add_pool(x, batch_idx)
        elif self.pooling == 'mean':
            x = geom_nn.global_mean_pool(x, batch_idx)
        elif self.pooling == 'max':
            x = geom_nn.global_max_pool(x, batch_idx)
        x = F.relu(self.linear_1(x))
        out = self.linear_2(x)
        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device

