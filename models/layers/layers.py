import torch
import torch.nn as nn
import torch_geometric as tg
from torch_scatter import scatter_add, scatter

class FractalMP(tg.nn.MessagePassing):
    """Message Passing Neural Network Layer"""

    def __init__(
            self,
            node_features,
            edge_features,
            hidden_features,
            out_features,
            aggr="add",
            act=nn.ReLU,
            edge_inference=False,
    ):
        super().__init__(aggr=aggr)
        self.edge_inference = edge_inference
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_features),
            act(),
            nn.Linear(hidden_features, hidden_features),
            act(),
        )

        self.update_net = nn.Sequential(
            nn.Linear(node_features + hidden_features, hidden_features),
            act(),
            nn.Linear(hidden_features, out_features),
        )

        if edge_inference:
            self.edge_inferrer = nn.Sequential(
                nn.Linear(hidden_features, 1), nn.Sigmoid()
            )

    def forward(self, x, edge_index, subgraph_edge_index, node_subnode_index, subnode_node_index, ground_node,
                subgraph_batch_index, edge_attr=None):
        """Propagate"""
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = self.propagate(node_subnode_index, x=x, edge_attr=edge_attr)
        x = self.propagate(subgraph_edge_index, x=x, edge_attr=edge_attr)
        x = self.propagate(subnode_node_index, x=x, edge_attr=edge_attr)
        # global pool over nodes whose ground node is false
        # x[ground_node] = tg.nn.global_mean_pool(x[~ground_node], subgraph_batch_index)
        return x

    def message(self, x_i, x_j, edge_attr):
        """Send message with edge attributes"""
        input = [x_i, x_j, edge_attr]
        input = [val for val in input if val is not None]
        input = torch.cat(input, dim=-1)
        message = self.message_net(input)

        if self.edge_inference:
            message = message * self.edge_inferrer(message)
        return message

    def update(self, message, x):
        """Update node"""
        input = torch.cat((x, message), dim=-1)
        update = self.update_net(input)
        return update


class MP(tg.nn.MessagePassing):
    """Message Passing Neural Network Layer"""

    def __init__(
            self,
            node_features,
            edge_features,
            hidden_features,
            out_features,
            aggr="add",
            act=nn.ReLU,
            edge_inference=False,
    ):
        super().__init__(aggr=aggr)
        self.edge_inference = edge_inference
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_features),
            act(),
            nn.Linear(hidden_features, hidden_features),
            act(),
        )

        self.update_net = nn.Sequential(
            nn.Linear(node_features + hidden_features, hidden_features),
            act(),
            nn.Linear(hidden_features, out_features),
        )

        if edge_inference:
            self.edge_inferrer = nn.Sequential(
                nn.Linear(hidden_features, 1), nn.Sigmoid()
            )

    def forward(self, x, edge_index, edge_attr=None):
        """Propagate"""
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # global pool over nodes whose ground node is false
        # x[ground_node] = tg.nn.global_mean_pool(x[~ground_node], subgraph_batch_index)
        return x

    def message(self, x_i, x_j, edge_attr):
        """Send message with edge attributes"""
        input = [x_i, x_j, edge_attr]
        input = [val for val in input if val is not None]
        input = torch.cat(input, dim=-1)
        message = self.message_net(input)

        if self.edge_inference:
            message = message * self.edge_inferrer(message)
        return message

    def update(self, message, x):
        """Update node"""
        input = torch.cat((x, message), dim=-1)
        update = self.update_net(input)
        return update


class SimpleMP(tg.nn.MessagePassing):
    """Message Passing Neural Network Layer"""

    def __init__(
            self,
            node_features,
            edge_features,
            hidden_features,
            out_features,
            aggr="add",
            act=nn.ReLU,
            edge_inference=False,
    ):
        super().__init__(aggr=aggr)
        self.edge_inference = edge_inference
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_features),
            act(),
            nn.Linear(hidden_features, hidden_features),
            act(),
        )

        if edge_inference:
            self.edge_inferrer = nn.Sequential(
                nn.Linear(hidden_features, 1), nn.Sigmoid()
            )

    def forward(self, x, edge_index, edge_attr=None):
        """Propagate"""
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x

    def message(self, x_i, x_j, edge_attr):
        """Send message with edge attributes"""
        input = [x_i, x_j, edge_attr]
        input = [val for val in input if val is not None]
        input = torch.cat(input, dim=-1)
        message = self.message_net(input)

        if self.edge_inference:
            message = message * self.edge_inferrer(message)
        return message

    def update(self, message, x):
        """Update node"""
        return x if message is None else message

class EGNN_FullLayer(tg.nn.MessagePassing):
    def __init__(self, emb_dim, activation="relu", norm="layer", aggr="add"):
        """E(n) Equivariant GNN Layer

        Paper: E(n) Equivariant Graph Neural Networks, Satorras et al.

        Args:
            emb_dim: (int) - hidden dimension `d`
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.activation = {"swish": nn.SiLU(), "relu": nn.ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d}[norm]

        # MLP `\psi_h` for computing messages `m_ij`
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * emb_dim + 1, emb_dim),
            self.norm(emb_dim),
            self.activation,
            nn.Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )
        # MLP `\psi_x` for computing messages `\overrightarrow{m}_ij`
        self.mlp_pos = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), self.norm(emb_dim), self.activation, nn.Linear(emb_dim, 1)
        )
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            nn.Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )

    def forward(self, h, pos, edge_index):
        """
        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
        Returns:
            out: [(n, d),(n,3)] - updated node features
        """
        out = self.propagate(edge_index, h=h, pos=pos)
        return out

    def message(self, h_i, h_j, pos_i, pos_j):
        # Compute messages
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1).unsqueeze(1)
        msg = torch.cat([h_i, h_j, dists], dim=-1)
        msg = self.mlp_msg(msg)
        # Scale magnitude of displacement vector
        pos_diff = pos_diff * self.mlp_pos(msg)  # torch.clamp(updates, min=-100, max=100)
        return msg, pos_diff

    def aggregate(self, inputs, index):
        msgs, pos_diffs = inputs
        # Aggregate messages
        msg_aggr = scatter(msgs, index, dim=self.node_dim, reduce=self.aggr)
        # Aggregate displacement vectors
        pos_aggr = scatter(pos_diffs, index, dim=self.node_dim, reduce="mean")
        return msg_aggr, pos_aggr

    def update(self, aggr_out, h, pos):
        msg_aggr, pos_aggr = aggr_out
        upd_out = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        upd_pos = pos + pos_aggr
        return upd_out, upd_pos

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"

class EGNNLayer(tg.nn.MessagePassing):
    def __init__(self, emb_dim, activation="relu", norm="layer", aggr="add"):
        """E(n) Equivariant GNN Layer

        Paper: E(n) Equivariant Graph Neural Networks, Satorras et al.

        Args:
            emb_dim: (int) - hidden dimension `d`
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.activation = {"swish": nn.SiLU(), "relu": nn.ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm,
                     "batch": torch.nn.BatchNorm1d,
                     "none": nn.Identity}[norm]

        # MLP `\psi_h` for computing messages `m_ij`
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * emb_dim + 1, emb_dim),
            self.norm(emb_dim),
            self.activation,
            nn.Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )

        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            nn.Linear(emb_dim, emb_dim),
            self.norm(emb_dim) if norm != "none" else nn.Identity(),
            self.activation,
        )

    def forward(self, h, pos, edge_index):
        """
        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
        Returns:
            out: [(n, d),(n,3)] - updated node features
        """
        out = self.propagate(edge_index, h=h, pos=pos)
        return out

    def message(self, h_i, h_j, pos_i, pos_j):
        # Compute messages
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1).unsqueeze(1)
        msg = torch.cat([h_i, h_j, dists], dim=-1)
        msg = self.mlp_msg(msg)
        # Scale magnitude of displacement vector
        return msg

    '''def aggregate(self, inputs, index):
        msgs = inputs
        # Aggregate messages
        msg_aggr = scatter(msgs, index, dim=self.node_dim, reduce=self.aggr)
        # Aggregate displacement vectors
        return msg_aggr'''

    def update(self, aggr_out, h, pos):
        msg_aggr = aggr_out
        upd_out = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        return upd_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"