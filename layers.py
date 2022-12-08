import torch
import torch.nn as nn
import torch_geometric as tg

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

    def forward(self, x, edge_index, subgraph_edge_index, node_subnode_index, subnode_node_index ,ground_node, subgraph_batch_index, edge_attr=None):
        """Propagate"""
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = self.propagate(node_subnode_index, x=x, edge_attr=edge_attr)
        x = self.propagate(subgraph_edge_index, x=x, edge_attr=edge_attr)
        x = self.propagate(subnode_node_index, x=x, edge_attr=edge_attr)
        # global pool over nodes whose ground node is false
        #x[ground_node] = tg.nn.global_mean_pool(x[~ground_node], subgraph_batch_index)
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
        #x[ground_node] = tg.nn.global_mean_pool(x[~ground_node], subgraph_batch_index)
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