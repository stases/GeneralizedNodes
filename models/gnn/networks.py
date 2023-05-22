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
                 residual=False, masking=False, layernorm=False, pool_all=False, **kwargs):
        super().__init__()
        self.name = 'FractalNet'
        self.depth = depth
        self.pool = pool
        self.add_residual_skip = residual
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

class EGNN_Full(nn.Module):
    def __init__(
            self,
            depth=5,
            hidden_features=128,
            node_features=1,
            out_features=1,
            activation="relu",
            norm="layer",
            aggr="sum",
            pool="add",
            residual=True,
            **kwargs
    ):
        """E(n) Equivariant GNN model 

        Args:
            depth: (int) - number of message passing layers
            hidden_features: (int) - hidden dimension
            node_features: (int) - initial node feature dimension
            out_features: (int) - output number of classes
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            pool: (str) - global pooling function (sum/mean)
            residual: (bool) - whether to use residual connections
        """
        super().__init__()
        # Name of the network
        self.name = "EGNN"

        # Embedding lookup for initial node features
        self.emb_in = nn.Linear(node_features, hidden_features)

        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(depth):
            self.convs.append(EGNNLayer(hidden_features, activation, norm, aggr))

        # Global pooling/readout function
        self.pool = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool}[pool]

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features)
        )
        self.residual = residual

    def forward(self, batch):

        h = self.emb_in(batch.x)  # (n,) -> (n, d)
        pos = batch.pos  # (n, 3)

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, batch.edge_index)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
            # pos = pos_update
            pos = pos

        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_features)

class EGNN(nn.Module):
    def __init__(
            self,
            depth,
            hidden_features,
            node_features,
            out_features,
            norm,
            activation="swish",
            aggr="sum",
            pool="add",
            residual=True,
            RFF_dim=None,
            RFF_sigma=None,
            **kwargs
    ):
        """E(n) Equivariant GNN model

        Args:
            depth: (int) - number of message passing layers
            hidden_features: (int) - hidden dimension
            node_features: (int) - initial node feature dimension
            out_features: (int) - output number of classes
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            pool: (str) - global pooling function (sum/mean)
            residual: (bool) - whether to use residual connections
        """
        super().__init__()
        # Name of the network
        self.name = "EGNN"

        # Embedding lookup for initial node features
        self.emb_in = nn.Linear(node_features, hidden_features)

        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(depth):
            self.convs.append(EGNNLayer(hidden_features, activation, norm, aggr, RFF_dim, RFF_sigma))

        # Global pooling/readout function
        self.pool = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool}[pool]

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features)
        )
        self.residual = residual

    def forward(self, batch):

        h = self.emb_in(batch.x)  # (n,) -> (n, d)
        pos = batch.pos  # (n, 3)

        for conv in self.convs:
            # Message passing layer
            h_update = conv(h, pos, batch.edge_index)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update

            # Update node coordinates (no residual) (n, 3) -> (n, 3)


        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_features)

class Fractal_EGNN(nn.Module):
    def __init__(
            self,
            depth=5,
            hidden_features=128,
            node_features=1,
            out_features=1,
            activation="swish",
            norm="layer",
            aggr="sum",
            pool="add",
            residual=True,
            RFF_dim=None,
            RFF_sigma=None,
            mask=None,
            **kwargs
    ):
        """E(n) Equivariant GNN model

        Args:
            depth: (int) - number of message passing layers
            hidden_features: (int) - hidden dimension
            node_features: (int) - initial node feature dimension
            out_features: (int) - output number of classes
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            pool: (str) - global pooling function (sum/mean)
            residual: (bool) - whether to use residual connections
        """
        super().__init__()
        # Name of the network
        self.name = "Fractal_EGNN"
        self.depth = depth
        self.mask = mask
        # Embedding lookup for initial node features
        self.emb_in = nn.Linear(node_features, hidden_features)

        # Stack of GNN layers
        self.ground_mps = torch.nn.ModuleList()
        self.ground_to_sub_mps = torch.nn.ModuleList()
        self.sub_mps = torch.nn.ModuleList()
        self.sub_to_ground_mps = torch.nn.ModuleList()
        for layer in range(depth):
            #self.convs.append(EGNNLayer(hidden_features, activation, norm, aggr))
            self.ground_mps.append(EGNNLayer(hidden_features, activation, norm, aggr, RFF_dim, RFF_sigma))
            self.ground_to_sub_mps.append(EGNNLayer(hidden_features, activation, norm, aggr, RFF_dim, RFF_sigma))
            self.sub_mps.append(EGNNLayer(hidden_features, activation, norm, aggr, RFF_dim, RFF_sigma))
            self.sub_to_ground_mps.append(EGNNLayer(hidden_features, activation, norm, aggr, RFF_dim, RFF_sigma))

        # Global pooling/readout function
        self.pool = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool}[pool]

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features)
        )
        self.residual = residual

    def forward(self, batch):
        num_nodes = batch.x.shape[0]
        device = batch.x.device
        h = self.emb_in(batch.x)  # (n,) -> (n, d)
        #print("H is: ", h)
        pos = batch.pos  # (n, 3)
        for layer_idx in range(self.depth):
            # Ground node message passing layer
            h_0 = h
            mask = catch_lone_sender(batch.edge_index, num_nodes).to(device) if self.mask else None
            h = self.ground_mps[layer_idx](h, pos, batch.edge_index, mask)
            if self.residual:
                h = h + h_0

            # Ground to subnode message passing layer
            h_0 = h
            mask = catch_lone_sender(batch.node_subnode_index, num_nodes).to(device) if self.mask else None
            h = self.ground_to_sub_mps[layer_idx](h, pos, batch.node_subnode_index, mask)
            if self.residual:
                h = h + h_0

            # Subnode message passing layer
            h_0 = h
            mask = catch_lone_sender(batch.subgraph_edge_index, num_nodes).to(device) if self.mask else None
            h = self.sub_mps[layer_idx](h, pos, batch.subgraph_edge_index, mask)
            if self.residual:
                h = h + h_0

            # Subnode to ground node message passing layer
            h_0 = h
            mask = catch_lone_sender(batch.subnode_node_index, num_nodes).to(device) if self.mask else None
            h = self.sub_to_ground_mps[layer_idx](h, pos, batch.subnode_node_index, mask)
            if self.residual:
                h = h + h_0
            # Update node features (n, d) -> (n, d)
        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_features)

class Fractal_EGNN_v2(nn.Module):
    def __init__(
            self,
            depth=5,
            hidden_features=128,
            node_features=1,
            out_features=1,
            activation="swish",
            norm="layer",
            aggr="sum",
            pool="add",
            residual=True,
            RFF_dim=None,
            RFF_sigma=None,
            mask=None,
            **kwargs
    ):
        """E(n) Equivariant GNN model

        Args:
            depth: (int) - number of message passing layers
            hidden_features: (int) - hidden dimension
            node_features: (int) - initial node feature dimension
            out_features: (int) - output number of classes
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            pool: (str) - global pooling function (sum/mean)
            residual: (bool) - whether to use residual connections
        """
        super().__init__()
        # Name of the network
        self.name = "Fractal_EGNN_2"
        self.depth = depth
        self.mask = mask
        # Embedding lookup for initial node features
        self.emb_in = nn.Linear(node_features, hidden_features)

        # Stack of GNN layers
        self.ground_mps = torch.nn.ModuleList()
        self.ground_to_sub_mps = torch.nn.ModuleList()
        self.sub_mps = torch.nn.ModuleList()
        self.sub_to_ground_mps = torch.nn.ModuleList()
        for layer in range(depth):
            #self.convs.append(EGNNLayer(hidden_features, activation, norm, aggr))
            self.ground_mps.append(EGNNLayer(hidden_features, activation, norm, aggr, RFF_dim, RFF_sigma))
            self.ground_to_sub_mps.append(EGNNLayer(hidden_features, activation, norm, aggr, RFF_dim, RFF_sigma))
            self.sub_mps.append(EGNNLayer(hidden_features, activation, norm, aggr, RFF_dim, RFF_sigma))
            self.sub_to_ground_mps.append(EGNNLayer(hidden_features, activation, norm, aggr, RFF_dim, RFF_sigma))

        # Global pooling/readout function
        self.pool = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool}[pool]

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features)
        )
        self.residual = residual

    def forward(self, batch):
        num_nodes = batch.x.shape[0]
        device = batch.x.device
        h = self.emb_in(batch.x)  # (n,) -> (n, d)
        #print("H is: ", h)
        pos = batch.pos  # (n, 3)
        for layer_idx in range(self.depth):
            # Residual connection
            h_0 = h

            # Ground node message passing layer
            mask = catch_lone_sender(batch.edge_index, num_nodes).to(device) if self.mask else None
            h = self.ground_mps[layer_idx](h, pos, batch.edge_index, mask=mask)

            # Ground to subnode message passing layer
            mask = catch_lone_sender(batch.node_subnode_index, num_nodes).to(device) if self.mask else None
            h = self.ground_to_sub_mps[layer_idx](h, pos, batch.node_subnode_index, mask=mask)

            # Subnode message passing layer
            mask = catch_lone_sender(batch.subgraph_edge_index, num_nodes).to(device) if self.mask else None
            h = self.sub_mps[layer_idx](h, pos, batch.subgraph_edge_index, mask=mask)

            # Subnode to ground node message passing layer
            mask = catch_lone_sender(batch.subnode_node_index, num_nodes).to(device) if self.mask else None
            h = self.sub_to_ground_mps[layer_idx](h, pos, batch.subnode_node_index, mask=mask)

            # Adding residual connections at the very end
            if self.residual:
                h = h + h_0
            # Update node features (n, d) -> (n, d)
        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_features)