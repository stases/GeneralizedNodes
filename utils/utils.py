import torch
from torch_geometric.utils import to_undirected
def catch_lone_sender(edge_index, num_nodes):
    receiver = edge_index[1]
    is_receiver = torch.zeros(num_nodes, dtype=torch.bool)
    is_receiver[receiver] = True
    return is_receiver
def fully_connected_edge_index(num_nodes):
    row = torch.arange(num_nodes)
    col = torch.arange(num_nodes)
    edge_index = torch.stack([row, col], dim=0)
    edge_index = to_undirected(edge_index)
    return edge_index