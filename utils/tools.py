import torch
from torch_geometric.utils import to_undirected
def catch_lone_sender(edge_index, num_nodes):
    receiver = edge_index[1]
    is_receiver = torch.zeros(num_nodes, dtype=torch.bool)
    is_receiver[receiver] = True
    return is_receiver
def fully_connected_edge_index(num_nodes):
    # Create a tensor with all possible combinations of node indices
    all_edges = torch.combinations(torch.arange(num_nodes), r=2)

    # Create a tensor for the reverse edges (since it's an undirected graph)
    reverse_edges = all_edges.flip(1)

    # Concatenate both sets of edges to get a fully connected edge index
    edge_index = torch.cat((all_edges, reverse_edges), dim=0).t().contiguous()

    return edge_index

def fully_connect_existing_nodes(edge_index):
    # Get the device
    device = edge_index.device

    # Find the minimum and maximum node indices in the existing edge index
    min_node = edge_index.min().item()
    max_node = edge_index.max().item()

    # Create a tensor with all possible combinations of node indices between min_node and max_node
    all_nodes = torch.arange(min_node, max_node + 1)
    all_edges = torch.combinations(all_nodes, r=2)

    # Create a tensor for the reverse edges (since it's an undirected graph)
    reverse_edges = all_edges.flip(1)

    # Concatenate both sets of edges to get a fully connected edge index
    new_edge_index = torch.cat((all_edges, reverse_edges), dim=0).t().contiguous()
    new_edge_index = new_edge_index.to(device)

    # Remove any duplicate edges that may already be present in the original edge index
    combined_edges = torch.cat((edge_index, new_edge_index), dim=1)
    unique_edges, _ = combined_edges.unique(dim=1, return_inverse=True)

    return unique_edges

def compute_mean_mad(train_loader, label_property):
    values = train_loader.dataset.data.y[:, label_property]
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad