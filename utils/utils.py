import torch

def catch_lone_sender(edge_index, num_nodes):
    receiver = edge_index[1]
    is_receiver = torch.zeros(num_nodes, dtype=torch.bool)
    is_receiver[receiver] = True
    return is_receiver