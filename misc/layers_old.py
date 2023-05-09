class EGNNLayer(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.message_mlp = nn.Sequential(nn.Linear(2 * num_hidden + 1, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden))
        self.update_mlp = nn.Sequential(nn.Linear(2 * num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden))

    def forward(self, x, pos, edge_index):
        send, rec = edge_index
        state = torch.cat((x[send], x[rec], torch.linalg.norm(pos[send] - pos[rec], dim=1).unsqueeze(1)), dim=1)
        message = self.message_mlp(state)
        aggr = scatter_add(message, rec, dim=0)
        update = self.update_mlp(torch.cat((x, aggr), dim=1))
        return update
