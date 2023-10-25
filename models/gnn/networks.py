import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as geom_nn
from torch_geometric.nn.conv import TransformerConv, GCNConv, RGCNConv
from utils.tools import catch_lone_sender, fully_connected_edge_index
from ..layers.layers import FractalMP, MP, EGNNLayer, MultiHeadGATLayer, EGNN_FullLayer, MPNNLayer, RelEGNNLayer

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

class MPNN(nn.Module):
    """ Message Passing Neural Network """

    def __init__(self, node_features, edge_features, hidden_features, out_features, depth, aggr="mean",
                 act=nn.ReLU, pool="add", dropout=0.3, **kwargs):
        super().__init__()
        num_layers = depth
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.embedder = nn.Sequential(nn.Linear(node_features, hidden_features),
                                      act(),
                                      nn.Linear(hidden_features, hidden_features))

        layers = []
        for i in range(num_layers):
            layer = MPNNLayer(node_features=hidden_features,
                              hidden_features=hidden_features,
                              edge_features=edge_features,
                              out_features=hidden_features,
                              aggr=aggr,
                              act=act
                              )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self.pooler = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool, "none": None}[pool]

        self.head = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                  act(),
                                  nn.Linear(hidden_features, out_features))

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        x = self.embedder(x)
        x = self.dropout(x)  # Apply dropout to node features

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = self.dropout(x)  # Apply dropout after each layer

        if self.pooler:
            x = self.pooler(x, batch.batch)

        x = self.head(x)
        return x

class RCGNN(nn.Module):
    """ Message Passing Neural Network """

    def __init__(self, node_features, edge_features, hidden_features, out_features, depth, aggr="mean",
                 act=nn.ReLU, pool="add", dropout=0.3, no_relations=False, **kwargs):
        super().__init__()
        num_layers = depth
        self.no_relations = no_relations
        print("No relations: ", self.no_relations)
        if self.no_relations:
            num_relations = 1
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.embedder = nn.Sequential(nn.Linear(node_features, hidden_features),
                                      act(),
                                      nn.Linear(hidden_features, hidden_features))
        self.activation = act()  # Activation function for use after RGCNConv layers

        layers = []
        for i in range(num_layers):
            layer = RGCNConv(hidden_features, hidden_features, num_relations=edge_features)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self.pooler = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool, "none": None}[pool]

        self.head = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                  act(),
                                  nn.Linear(hidden_features, out_features))

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr.argmax(dim=-1)
        if self.no_relations:
            edge_attr = torch.zeros(edge_index.shape[1], dtype=torch.long, device=x.device)
        x = self.embedder(x)
        x = self.dropout(x)  # Apply dropout to node features

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            # Add ReLU activation for all layers except the last one
            if i != len(self.layers) - 1:
                x = self.activation(x)
            x = self.dropout(x)  # Apply dropout after each layer

        if self.pooler:
            x = self.pooler(x, batch.batch)

        x = self.head(x)
        return x

class Transformer_MPNN(nn.Module):
    """ Message Passing Neural Network """

    def __init__(self,
                 node_features,
                 edge_features,
                 hidden_features,
                 out_features,
                 depth,
                 ascend_depth,
                 num_heads,
                 num_ascend_heads,
                 mask=None,
                 residual=True,
                 norm="layer",
                 aggr="mean",
                 act=nn.ReLU,
                 pool="add",
                 only_ground=False,
                 only_sub=False,
                 **kwargs):
        """
        Here we choose global_add_pool as our default graph pooling methods,
        but with the other type of tasks, make sure to try also pooling methods like [global_max_pool, global_mean_pool]
        to make your network have specific features.
        """
        super().__init__()

        self.residual = residual
        self.mask = mask

        self.depth = depth
        self.ascend_depth = ascend_depth

        self.only_ground = only_ground
        self.only_sub = only_sub

        self.ground_mps = torch.nn.ModuleList()
        self.ground_to_sub_mps = torch.nn.ModuleList()
        self.sub_mps = torch.nn.ModuleList()
        self.sub_to_ground_mps = torch.nn.ModuleList()
        self.descend_normalization = torch.nn.ModuleList()
        self.sub_normalization = torch.nn.ModuleList()
        self.ascend_normalization = torch.nn.ModuleList()

        self.embedder = nn.Sequential(nn.Linear(node_features, hidden_features),
                                      act(),
                                      nn.Linear(hidden_features, hidden_features))

        for layer in range(depth):
            self.ground_mps.append(MPNNLayer(node_features=hidden_features,
                              hidden_features=hidden_features,
                              edge_features=edge_features,
                              out_features=hidden_features,
                              aggr=aggr,
                              act=act
                              ))
            self.ground_to_sub_mps.append(TransformerConv(hidden_features, hidden_features, num_heads, concat=False))
            self.descend_normalization.append(nn.LayerNorm(hidden_features)) if norm == "layer" else nn.Identity()
            self.sub_mps.append(TransformerConv(hidden_features, hidden_features, num_heads, concat=False))
            self.sub_normalization.append(nn.LayerNorm(hidden_features)) if norm == "layer" else nn.Identity()

        for layer in range(ascend_depth):
            self.sub_to_ground_mps.append(TransformerConv(hidden_features, hidden_features, num_ascend_heads, concat=False))
            self.ascend_normalization.append(nn.LayerNorm(hidden_features)) if norm == "layer" else nn.Identity()

        self.pool = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool}[pool]

        self.head = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                  act(),
                                  nn.Linear(hidden_features, out_features))

    def forward(self, batch):
        num_nodes = batch.x.shape[0]
        x = batch.x
        device = x.device
        h = self.embedder(x)

        for layer_idx in range(self.depth):
            # Residual connection
            h_old = h.clone()
            h_0 = h
            # Ground node message passing layer
            mask = catch_lone_sender(batch.edge_index, num_nodes).to(device) if self.mask else None
            h = self.ground_mps[layer_idx](h, batch.edge_index)
            if self.residual:
                h = h + h_0
            if self.mask:
                h[~batch.ground_node] = h_old[~batch.ground_node]

            # Ground to subnode message passing layer
            h_old = h.clone()
            h_0 = h
            h = self.ground_to_sub_mps[layer_idx](h, batch.node_subnode_index)
            if self.residual:
                h = h + h_0
            h = self.descend_normalization[layer_idx](h)
            if self.mask:
                h[batch.ground_node] = h_old[batch.ground_node]

            # Subnode message passing layer
            h_old = h.clone()
            h_0 = h
            h = self.sub_mps[layer_idx](h, batch.subgraph_edge_index)
            if self.residual:
                h = h + h_0
            h = self.sub_normalization[layer_idx](h)
            if self.mask:
                h[batch.ground_node] = h_old[batch.ground_node]

        for layer_idx in range(self.ascend_depth):
            h_0 = h
            h = self.sub_to_ground_mps[layer_idx](h, batch.subnode_node_index)
            if self.residual:
                h = h + h_0
            h = self.ascend_normalization[layer_idx](h)

        if self.only_ground and self.only_sub:
            h = self.pool(h, batch.batch)
            h = self.head(h)
            return h
        if self.only_ground:
            h = self.pool(h[batch.ground_node], batch.batch[batch.ground_node])
            h = self.head(h)
            return h
        elif self.only_sub:
            h = self.pool(h[~batch.ground_node], batch.batch[~batch.ground_node])
            h = self.head(h)
            return h
        else:
            h = self.pool(h, batch.batch)
            h = self.head(h)
            return h

class Simple_MPNN(nn.Module):
    """ Message Passing Neural Network """

    def __init__(self, node_features, hidden_features, out_features, depth, aggr="mean",
                 act=nn.ReLU, pool="add", dropout=0.3, **kwargs):
        super().__init__()
        self.name = 'Simple_Transformer_MPNN'
        num_layers = depth
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.embedder = nn.Identity()
        self.act = act()
        layers = []
        for i in range(num_layers):
            layer = GCNConv(node_features, hidden_features)
            layers.append(layer)
            node_features = hidden_features
        self.layers = nn.ModuleList(layers)

        self.pooler = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool, "none": None}[pool]

        self.head = nn.Linear(hidden_features, out_features)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        x = self.embedder(x)
        x = self.dropout(x)  # Apply dropout to node features

        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)  # Apply dropout after each layer

        if self.pooler:
            x = self.pooler(x, batch.batch)

        x = self.head(x)
        return x

class Simple_Transformer_MPNN(nn.Module):
    """ Message Passing Neural Network """

    def __init__(self, node_features, edge_features, hidden_features, out_features, depth, aggr="mean",
                 act=nn.ReLU, pool="add", dropout=0.3, mask=True, **kwargs):
        super().__init__()
        self.name = 'Simple_MPNN'
        num_layers = depth
        self.depth = depth
        self.mask = mask
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.embedder = nn.Linear(node_features, hidden_features)
        self.activation = act()
        print('Dropout is: ', dropout)
        print("Model name is: ", self.name)
        self.ground_mps = []
        self.ground_to_sub_mps = []
        self.sub_mps = []
        self.sub_to_ground_mps = []
        for i in range(num_layers):
            #layer = GCNConv(node_features, hidden_features)
            #self.ground_mps.append(GCNConv(hidden_features, hidden_features))
            #self.ground_to_sub_mps.append(GCNConv(hidden_features, hidden_features))
            #self.sub_mps.append(GCNConv(hidden_features, hidden_features))
            #self.sub_to_ground_mps.append(GCNConv(hidden_features, hidden_features))
            pass;
        self.ground_mps = nn.ModuleList([GCNConv(hidden_features, hidden_features) for _ in range(num_layers)])
        self.ground_to_sub_mps = nn.ModuleList([GCNConv(hidden_features, hidden_features) for _ in range(num_layers)])
        self.sub_mps = nn.ModuleList([GCNConv(hidden_features, hidden_features) for _ in range(num_layers)])
        self.sub_to_ground_mps = nn.ModuleList([GCNConv(hidden_features, hidden_features) for _ in range(num_layers)])

        self.pooler = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool, "none": None}[pool]

        self.head = nn.Linear(hidden_features, out_features)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        x = self.embedder(x)
        x = self.dropout(x)  # Apply dropout to node features

        for layer_idx in range(self.depth):
            x_backup = x[~batch.ground_node]
            x = self.ground_mps[layer_idx](x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)  # Apply dropout after each layer
            x[~batch.ground_node] = x_backup

            x_backup = x[batch.ground_node]
            x = self.ground_to_sub_mps[layer_idx](x, batch.node_subnode_index)
            x = self.activation(x)
            x = self.dropout(x)  # Apply dropout after each layer
            x[batch.ground_node] = x_backup

            x_backup = x[batch.ground_node]
            x = self.sub_mps[layer_idx](x, batch.subgraph_edge_index)
            x = self.activation(x)
            x = self.dropout(x)  # Apply dropout after each layer
            x[batch.ground_node] = x_backup

            x_backup = x[~batch.ground_node]
            x = self.sub_to_ground_mps[layer_idx](x, batch.subnode_node_index)
            x = self.activation(x)
            x = self.dropout(x)  # Apply dropout after each layer
            x[~batch.ground_node] = x_backup

        if self.pooler:
            x = self.pooler(x, batch.batch)

        x = self.head(x)
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
            return_pos=False,
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
        self.name = "EGNN_Full"

        # Embedding lookup for initial node features
        self.emb_in = nn.Linear(node_features, hidden_features)

        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(depth):
            self.convs.append(EGNN_FullLayer(hidden_features, activation, norm, aggr))

        # Global pooling/readout function
        self.pool = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool, "none": None}[pool]

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features)
        )
        self.residual = residual
        self.return_pos = return_pos
	
    def forward(self, batch):
        h = self.emb_in(batch.x)  # (n,) -> (n, d)
        pos = batch.pos.clone()  # (n, 3)

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, batch.edge_index)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
            pos = pos_update
        if self.pool is not None:
            out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        else:
            out = h

        if self.return_pos:
            return pos, self.pred(out)
        else:
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
            return_pos=False,
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
        self.pool = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool, "none": None}[pool]

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features)
        )
        self.residual = residual

    def forward(self, batch):
        h = self.emb_in(batch.x)  # (n,) -> (n, d)
        #pos = batch.pos  # (n, 3)
        pos = batch.pos.clone()
        for conv in self.convs:
            # Message passing layer
            h_update = conv(h, pos, batch.edge_index)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
        out = h
        if self.pool is not None:
            out = self.pool(h, batch.batch)

        
        return self.pred(out)  # (batch_size, out_features)

class RelEGNN(nn.Module):
    def __init__(
            self,
            depth,
            hidden_features,
            node_features,
            out_features,
            num_relations,
            norm,
            activation="swish",
            aggr="sum",
            pool="add",
            residual=True,
            RFF_dim=None,
            RFF_sigma=None,
            return_pos=False,
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
        self.name = "RelEGNN"

        # Embedding lookup for initial node features
        self.emb_in = nn.Linear(node_features, hidden_features)

        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(depth):
            self.convs.append(RelEGNNLayer(hidden_features, num_relations, activation, norm, aggr, RFF_dim, RFF_sigma))

        # Global pooling/readout function
        self.pool = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool, "none": None}[pool]

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features)
        )
        self.residual = residual

    def forward(self, batch):
        h = self.emb_in(batch.x)  # (n,) -> (n, d)
        # pos = batch.pos  # (n, 3)
        pos = batch.pos.clone()
        edge_type = batch.edge_attr.argmax(dim=-1)
        for conv in self.convs:
            # Message passing layer
            h_update = conv(h, pos, batch.edge_index, edge_type)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
        out = h
        if self.pool is not None:
            out = self.pool(h, batch.batch)

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
            only_ground=False,
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
        self.only_ground = only_ground
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
        if self.only_ground:
            h = h[batch.ground_node]
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
            sub_aggr="sum",
            pool="add",
            residual=True,
            RFF_dim=None,
            RFF_sigma=None,
            mask=None,
            only_ground=False,
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
        self.only_ground = only_ground
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
            self.sub_mps.append(EGNNLayer(hidden_features, activation, norm, sub_aggr, RFF_dim, RFF_sigma))
            self.sub_to_ground_mps.append(EGNNLayer(hidden_features, activation, norm, sub_aggr, RFF_dim, RFF_sigma))

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
        if self.only_ground:
            h = h[batch.ground_node]
            batch.batch = batch.batch[batch.ground_node]
        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_features)

class Transformer_EGNN(nn.Module):
    def __init__(
            self,
            depth=5,
            hidden_features=128,
            node_features=1,
            out_features=1,
            num_heads=1,
            activation="swish",
            norm="layer",
            aggr="sum",
            sub_aggr="sum",
            pool="add",
            residual=True,
            RFF_dim=None,
            RFF_sigma=None,
            mask=None,
            only_ground=False,
            **kwargs
    ):
        super().__init__()
        # Name of the network
        self.name = "Transformer_EGNN"
        self.depth = depth
        self.mask = mask
        self.only_ground = only_ground
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
            self.ground_to_sub_mps.append(TransformerConv(hidden_features, hidden_features, num_heads, concat=False))
            self.sub_mps.append(TransformerConv(hidden_features, hidden_features, num_heads, concat=False))
            self.sub_to_ground_mps.append(TransformerConv(hidden_features, hidden_features, num_heads, concat=False))

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
        pos = batch.pos  # (n, 3)
        for layer_idx in range(self.depth):
            # Residual connection
            h_0 = h

            # Ground node message passing layer
            mask = catch_lone_sender(batch.edge_index, num_nodes).to(device) if self.mask else None
            h = self.ground_mps[layer_idx](h, pos, batch.edge_index, mask=mask)

            # Ground to subnode message passing layer
            mask = catch_lone_sender(batch.node_subnode_index, num_nodes).to(device) if self.mask else None
            h = self.ground_to_sub_mps[layer_idx](h, batch.node_subnode_index)

            # Subnode message passing layer
            mask = catch_lone_sender(batch.subgraph_edge_index, num_nodes).to(device) if self.mask else None
            h = self.sub_mps[layer_idx](h, batch.subgraph_edge_index)

            # Subnode to ground node message passing layer
            mask = catch_lone_sender(batch.subnode_node_index, num_nodes).to(device) if self.mask else None
            h = self.sub_to_ground_mps[layer_idx](h, batch.subnode_node_index)

            # Adding residual connections at the very end
            if self.residual:
                h = h + h_0
            # Update node features (n, d) -> (n, d)
        if self.only_ground:
            h = h[batch.ground_node]
            batch.batch = batch.batch[batch.ground_node]
        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_features)

class Transformer_EGNN_v2(nn.Module):
    def __init__(
            self,
            depth=5,
            ascend_depth=1,
            hidden_features=128,
            node_features=1,
            out_features=1,
            num_heads=1,
            num_ascend_heads=4,
            activation="swish",
            norm="layer",
            aggr="sum",
            sub_aggr="sum",
            pool="add",
            residual=True,
            RFF_dim=None,
            RFF_sigma=None,
            mask=None,
            only_ground=False,
            only_sub=False,
            **kwargs
    ):
        super().__init__()
        # Name of the network
        self.name = "Transformer_EGNN_v2"
        self.depth = depth
        self.ascend_depth = ascend_depth
        self.mask = mask
        self.only_ground = only_ground
        self.only_sub = only_sub
        # print only sub
        print("Only sub is {}".format(self.only_sub))
        # Embedding lookup for initial node features
        self.emb_in = nn.Linear(node_features, hidden_features)

        # Stack of GNN layers
        self.ground_mps = torch.nn.ModuleList()
        self.ground_to_sub_mps = torch.nn.ModuleList()
        self.sub_mps = torch.nn.ModuleList()
        self.sub_to_ground_mps = torch.nn.ModuleList()
        self.descend_normalization = torch.nn.ModuleList()
        self.sub_normalization = torch.nn.ModuleList()
        self.ascend_normalization = torch.nn.ModuleList()
        for layer in range(depth):
            self.ground_mps.append(EGNNLayer(hidden_features, activation, norm, aggr, RFF_dim, RFF_sigma))
            self.ground_to_sub_mps.append(TransformerConv(hidden_features, hidden_features, num_heads, concat=False))
            self.descend_normalization.append(nn.LayerNorm(hidden_features)) if norm == "layer" else nn.Identity()
            self.sub_mps.append(TransformerConv(hidden_features, hidden_features, num_heads, concat=False))
            self.sub_normalization.append(nn.LayerNorm(hidden_features)) if norm == "layer" else nn.Identity()
        for layer in range(ascend_depth):
            self.sub_to_ground_mps.append(TransformerConv(hidden_features, hidden_features, num_ascend_heads, concat=False))
            self.ascend_normalization.append(nn.LayerNorm(hidden_features)) if norm == "layer" else nn.Identity()
        # Global pooling/readout function
        self.pool = {"mean": tg.nn.global_mean_pool, "add": tg.nn.global_add_pool}[pool]

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_features*1, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features)
        )
        self.residual = residual

    def forward(self, batch):
        num_nodes = batch.x.shape[0]
        device = batch.x.device
        h = self.emb_in(batch.x)  # (n,) -> (n, d)
        pos = batch.pos  # (n, 3)
        for layer_idx in range(self.depth):
            # Residual connection
            h_0 = h
            h_before = h[batch.ground_node]
            # Ground node message passing layer
            mask = catch_lone_sender(batch.edge_index, num_nodes).to(device) if self.mask else None
            h = self.ground_mps[layer_idx](h, pos, batch.edge_index, mask=mask)
            if self.residual:
                h = h + h_0

            # Ground to subnode message passing layer
            h_old = h.clone()
            h_0 = h
            h = self.ground_to_sub_mps[layer_idx](h, batch.node_subnode_index)
            if self.residual:
                h = h + h_0
            h = self.descend_normalization[layer_idx](h)
            if self.mask:
                h[batch.ground_node] = h_old[batch.ground_node]

            # Subnode message passing layer
            h_old = h.clone()
            h_0 = h
            h = self.sub_mps[layer_idx](h, batch.subgraph_edge_index)
            if self.residual:
                h = h + h_0
            h = self.sub_normalization[layer_idx](h)
            if self.mask:
                h[batch.ground_node] = h_old[batch.ground_node]

        for layer_idx in range(self.ascend_depth):
            h_0 = h
            h = self.sub_to_ground_mps[layer_idx](h, batch.subnode_node_index)
            if self.residual:
                h = h + h_0
            h = self.ascend_normalization[layer_idx](h)

        if self.only_ground:
            out = self.pool(h[batch.ground_node], batch.batch[batch.ground_node])
        elif self.only_sub:
            out = self.pool(h[~batch.ground_node], batch.batch[~batch.ground_node])
        else:
            out = self.pool(h, batch.batch)
        # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_features)

class Superpixel_EGNN(nn.Module):
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
            mask=True,
            **kwargs
    ):
        super().__init__()
        # Name of the network
        self.name = "Superpixel_EGNN"
        self.depth = depth
        # Embedding lookup for initial node features
        self.emb_in = nn.Linear(node_features, hidden_features)

        # Stack of GNN layers
        self.ground_mps = torch.nn.ModuleList()
        self.ground_to_sub_mps = torch.nn.ModuleList()
        self.sub_mps = torch.nn.ModuleList()
        self.sub_to_ground_mps = torch.nn.ModuleList()
        for layer in range(depth):
            self.ground_mps.append(EGNN_FullLayer(hidden_features, activation, norm, aggr))
            self.ground_to_sub_mps.append(EGNN_FullLayer(hidden_features, activation, norm, aggr))
            self.sub_mps.append(EGNN_FullLayer(hidden_features, activation, norm, aggr))
            #self.sub_to_ground_mps.append(EGNN_FullLayer(hidden_features, activation, norm, aggr))
        self.residual = residual
        self.mask = mask

        self.pred = torch.nn.Sequential(
        torch.nn.Linear(hidden_features*1, hidden_features),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_features, out_features)
        )
    def forward(self, batch):

        h = self.emb_in(batch.x)  # (n,) -> (n, d)
        pos = batch.pos.clone()  # (n, 3)
        pos[~batch.ground_node] += torch.randn_like(pos[~batch.ground_node]) * 0.01
        h_ground = h[batch.ground_node]
        pos_ground = pos[batch.ground_node]

        h_sub = h[~batch.ground_node]
        pos_sub = pos[~batch.ground_node]

        for layer_idx in range(self.depth):
            h_old = h.clone()
            h_0 = h
            pos_old = pos.clone()
            h_update, pos_update = self.ground_mps[layer_idx](h, pos, batch.edge_index)
            h = h + h_update if self.residual else h_update
            pos = pos_update
            if self.mask:
                pos[batch.ground_node] = pos_old[batch.ground_node]

            pos_old = pos.clone()
            h_update, pos_update = self.ground_to_sub_mps[layer_idx](h, pos, batch.node_subnode_index)
            h = h + h_update if self.residual else h_update
            pos = pos_update
            if self.mask:
                pos[batch.ground_node] = pos_old[batch.ground_node]
            pos_old = pos.clone()

            pos_before = pos.clone()
            h_update, pos_update = self.sub_mps[layer_idx](h, pos, batch.subgraph_edge_index)
            #print('pos update is', pos_update)
            #print('difference is', pos_update-pos)
            h = h + h_update if self.residual else h_update
            pos = pos_update
            #print('difference is', pos-pos_before)


            if self.mask:
                pass;
                #pos[batch.ground_node] = pos_old[batch.ground_node]


        h = self.pred(h)
        superpixel_pos = pos[~batch.ground_node]
        superpixel_h = h[~batch.ground_node]
        return superpixel_pos, superpixel_h
