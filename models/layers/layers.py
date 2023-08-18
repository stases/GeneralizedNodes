import torch
import torch.nn as nn
import torch_geometric as tg
#from torch_scatter import scatter_add, scatter
import torch.nn.functional as F
import math
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax
from torch_scatter import scatter

def catch_lone_sender(edge_index, num_nodes):
    receiver = edge_index[1]
    is_receiver = torch.zeros(num_nodes, dtype=torch.bool)
    is_receiver[receiver] = True
    return is_receiver
class RFF(nn.Module):
    def __init__(self, in_features, out_features, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.in_features = in_features
        self.out_features = out_features

        if out_features % 2 != 0:
            self.compensation = 1
        else:
            self.compensation = 0

        B = torch.randn(int(out_features / 2) + self.compensation, in_features) * sigma
        B /= math.sqrt(2)
        self.register_buffer("B", B)

    def forward(self, x):
        x = F.linear(x, self.B)
        x = torch.cat((x.sin(), x.cos()), dim=-1)
        if self.compensation:
            x = x[..., :-1]
        return x

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, sigma={}".format(
            self.in_features, self.out_features, self.sigma
        )
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
class EGNN_FullLayer_Dojo(tg.nn.MessagePassing):
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
        self.update_pos = True
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
        if self.update_pos:
            pos_aggr = scatter(pos_diffs, index, dim=self.node_dim, reduce="mean")

        nodes_to_upd = torch.unique(index)
        msg_aggr = msg_aggr[nodes_to_upd]

        if self.update_pos:
            pos_aggr = pos_aggr[nodes_to_upd]
        else:
            pos_aggr = None

        return msg_aggr, pos_aggr, nodes_to_upd

    def update(self, aggr_out, h, pos):
        msg_aggr, pos_aggr, nodes_to_upd = aggr_out

        upd_out = h
        upd_out[nodes_to_upd] = self.mlp_upd(torch.cat([h[nodes_to_upd], msg_aggr], dim=-1))
        if self.update_pos:
            upd_pos = pos
            #print('pos before is ', pos)
            #print('pos aggr is ', pos_aggr)
            upd_pos[nodes_to_upd] = pos[nodes_to_upd] + pos_aggr
            # print the difference
            #print('pos after is ', upd_pos)

        else:
            upd_pos = pos

        return upd_out, upd_pos

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"
class MPNNLayer(tg.nn.MessagePassing):
    """ Message Passing Layer """

    def __init__(self, node_features, edge_features, hidden_features, out_features, aggr="mean", act=nn.ReLU):
        super().__init__(aggr=aggr)

        self.message_net = nn.Sequential(nn.Linear(2 * node_features + edge_features, hidden_features),
                                         act(),
                                         nn.Linear(hidden_features, hidden_features))

        self.update_net = nn.Sequential(nn.Linear(node_features + hidden_features, hidden_features),
                                        act(),
                                        nn.Linear(hidden_features, out_features))

    def forward(self, x, edge_index, edge_attr=None):
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x

    def message(self, x_i, x_j, edge_attr):
        """ Construct messages between nodes """
        input = [x_i, x_j] if edge_attr is None else [x_i, x_j, edge_attr]
        input = torch.cat(input, dim=-1)
        message = self.message_net(input)
        return message

    def update(self, message, x):
        """ Update node features """
        input = torch.cat((x, message), dim=-1)
        update = self.update_net(input)
        return update
class EGNNLayer(tg.nn.MessagePassing):
    def __init__(self, emb_dim, activation="relu", norm="layer", aggr="add", RFF_dim=None, RFF_sigma=None, mask=None):
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
        self.RFF_dim = RFF_dim
        self.RFF_sigma = RFF_sigma
        self.mask = mask
        # MLP `\psi_h` for computing messages `m_ij`
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * emb_dim + 1 if self.RFF_dim is None else 2 * emb_dim + RFF_dim, emb_dim),
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
        if self.RFF_dim is not None:
            self.RFF = RFF(1, RFF_dim, RFF_sigma)

    def forward(self, h, pos, edge_index, mask=None):
        """
        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            mask: (n, d) - mask for node features
        Returns:
            out: [(n, d),(n,3)] - updated node features
        """
        self.mask = mask
        out = self.propagate(edge_index, h=h, pos=pos, mask=mask)
        return out

    def message(self, h_i, h_j, pos_i, pos_j):
        # Compute messages
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1).unsqueeze(1)
        if self.RFF_dim is not None:
            dists = self.RFF(dists)
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

    def update(self, aggr_out, h):
        msg_aggr = aggr_out
        upd_out = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        if self.mask is not None:
            upd_out = torch.where(self.mask.unsqueeze(-1), upd_out, h)
        return upd_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"

class RelEGNNLayer(tg.nn.MessagePassing):
    def __init__(self, emb_dim, num_relations, activation="relu", norm="layer", aggr="add", RFF_dim=None, RFF_sigma=None, mask=None):
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
        self.RFF_dim = RFF_dim
        self.RFF_sigma = RFF_sigma
        self.mask = mask
        # MLP `\psi_h` for computing messages `m_ij`
        self.mlps_msg = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * emb_dim + 1 if self.RFF_dim is None else 2 * emb_dim + RFF_dim, emb_dim),
                self.norm(emb_dim),
                self.activation,
                nn.Linear(emb_dim, emb_dim),
                self.norm(emb_dim),
                self.activation,
            )
            for _ in range(num_relations)
        ])

        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            nn.Linear(emb_dim, emb_dim),
            self.norm(emb_dim) if norm != "none" else nn.Identity(),
            self.activation,
        )
        if self.RFF_dim is not None:
            self.RFF = RFF(1, RFF_dim, RFF_sigma)

    def forward(self, h, pos, edge_index, edge_type, mask=None):
        """
        Args:
            # ... (rest of the arguments)
            edge_type: (e) - edge type for each edge in edge_index
        """
        self.mask = mask
        out = self.propagate(edge_index, h=h, pos=pos, edge_type=edge_type, mask=mask)
        return out

    def message(self, h_i, h_j, pos_i, pos_j, edge_type):
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1).unsqueeze(1)
        if self.RFF_dim is not None:
            dists = self.RFF(dists)
        msg_inputs = torch.cat([h_i, h_j, dists], dim=-1)

        # Vectorized application of the correct MLP based on edge type
        msg = torch.stack([mlp(msg_inputs) for mlp in self.mlps_msg], dim=0)  # Stack outputs
        msg = msg[edge_type, torch.arange(edge_type.size(0))]  # Select the right output for each edge

        return msg

    '''def aggregate(self, inputs, index):
        msgs = inputs
        # Aggregate messages
        msg_aggr = scatter(msgs, index, dim=self.node_dim, reduce=self.aggr)
        # Aggregate displacement vectors
        return msg_aggr'''

    def update(self, aggr_out, h):
        msg_aggr = aggr_out
        upd_out = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        if self.mask is not None:
            upd_out = torch.where(self.mask.unsqueeze(-1), upd_out, h)
        return upd_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"
class AbstractEGNNLayer(tg.nn.MessagePassing):
    def __init__(self, emb_dim, activation="relu", norm="layer", aggr="add",  mask=None):
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
        self.mask = mask
        # MLP `\psi_h` for computing messages `m_ij`
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * emb_dim , emb_dim),
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
    def forward(self, h, pos, edge_index, mask=None):
        """
        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            mask: (n, d) - mask for node features
        Returns:
            out: [(n, d),(n,3)] - updated node features
        """
        self.mask = mask
        out = self.propagate(edge_index, h=h, pos=pos, mask=mask)
        return out

    def message(self, h_i, h_j, pos_i, pos_j):
        # Compute messages
        msg = torch.cat([h_i, h_j], dim=-1)
        msg = self.mlp_msg(msg)
        # Scale magnitude of displacement vector
        return msg

    '''def aggregate(self, inputs, index):
        msgs = inputs
        # Aggregate messages
        msg_aggr = scatter(msgs, index, dim=self.node_dim, reduce=self.aggr)
        # Aggregate displacement vectors
        return msg_aggr'''

    def update(self, aggr_out, h):
        msg_aggr = aggr_out
        upd_out = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        if self.mask is not None:
            upd_out = torch.where(self.mask.unsqueeze(-1), upd_out, h)
        return upd_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"
class MultiHeadGATLayer(tg.nn.MessagePassing):
    def __init__(self, emb_dim, num_heads=1, activation="relu", norm="layer", aggr="add", mask=None):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.activation = {"swish": nn.SiLU(), "relu": nn.ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm,
                     "batch": torch.nn.BatchNorm1d,
                     "none": nn.Identity}[norm]
        self.mask = mask

        # Multi-head attention
        self.lin_l = nn.Linear(emb_dim, num_heads * emb_dim, bias=False)
        self.lin_r = nn.Linear(emb_dim, num_heads * emb_dim, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, num_heads, emb_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, num_heads, emb_dim))

        # Output layer
        self.lin_o = nn.Linear(emb_dim * num_heads, emb_dim)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)

    def forward(self, h, edge_index, mask=None):
        self.mask = mask

        h_l = self.lin_l(h).view(-1, self.num_heads, self.emb_dim)
        h_r = self.lin_r(h).view(-1, self.num_heads, self.emb_dim)

        return self.propagate(edge_index, h_l=h_l, h_r=h_r, mask=mask)

    def message(self, h_l_i, h_l_j, h_r_i, h_r_j, index):
        alpha_l = (h_l_j * self.att_l).sum(dim=-1)
        alpha_r = (h_r_i * self.att_r).sum(dim=-1)

        alpha = alpha_r + alpha_l
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, num_nodes=h_l_i.size(1))

        return alpha.view(-1, self.num_heads, 1) * h_r_j

    def update(self, aggr_out, h):
        msg_aggr = aggr_out
        upd_out = self.lin_o(msg_aggr)
        if self.mask is not None:
            upd_out = torch.where(self.mask.unsqueeze(-1), upd_out, h)
        return upd_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, num_heads={self.num_heads}, aggr={self.aggr})"
class TransformerConv(tg.nn.MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')