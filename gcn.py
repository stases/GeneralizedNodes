import torch
import torch.nn as nn
import torch_geometric as tg


class GCN(nn.Module):
    """A simple GCN, Message Passing Model"""

    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        out_features,
        num_layers,
        aggr="add",
        act=nn.ReLU,
        edge_inference=False,
        pool="add",
    ):
        super().__init__()

        # If the activation function is loaded from the yaml file, it will be a string
        # Need to fetch it from the torch.nn.__dict__
        if isinstance(act, str):
            act = getattr(nn, act)

        self.embedder = nn.Linear(node_features, hidden_features)
        layers = []
        for i in range(num_layers):
            layers.append(
                MessagePassingLayer(
                    hidden_features,
                    edge_features,
                    hidden_features,
                    hidden_features,
                    aggr,
                    act,
                    edge_inference,
                )
            )
        self.layers = nn.ModuleList(layers)

        if pool is None:
            self.pooler = pool
        elif pool == "add":
            self.pooler = tg.nn.global_add_pool
        elif pool == "mean":
            self.pooler = tg.nn.global_mean_pool
        elif pool == "no_pooling":
            self.pooler = nn.Identity()
        else:
            raise ValueError("Unknown pooling type")

        self.pre_pool = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            act(),
            nn.Linear(hidden_features, hidden_features),
        )

        self.post_pool = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            act(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.embedder(x)

        for layer in self.layers:
            x += layer(x, edge_index, edge_attr)  # Skip connection

        x = self.pre_pool(x)
        if self.pooler:
            x = self.pooler(x, batch)
        x = self.post_pool(x)
        return x


class MessagePassingLayer(tg.nn.MessagePassing):
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
