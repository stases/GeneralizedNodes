from models.layers.layers import MultiHeadGATLayer
import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Create a graph with random initial one-hot node features and some edge index
num_nodes = 10
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
x = torch.zeros(num_nodes, 9)
x[torch.randint(0, num_nodes, (1,)), :] = 1
# Create a MultiHeadGATLayer with 3 heads
layer = MultiHeadGATLayer(emb_dim=9, num_heads=3)
# Pass the graph through the layer
x = layer(x, edge_index)
# Print the output
print(x)
