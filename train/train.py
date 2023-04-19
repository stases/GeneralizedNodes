import torch
import numpy as np
from warnings import warn
from torch_geometric.datasets import QM9
from torch.autograd import gradcheck
import torch.nn as nn
from torch_geometric.loader import DataLoader
from models.gnn.networks import FractalNet
from utils.subgraph import Graph_to_Subgraph

def get_qm9(data_dir, device="cuda", LABEL_INDEX = 7, transform=None):
    """Download the QM9 dataset from pytorch geometric. Put it onto the device. Split it up into train / validation / test.
    Args:
        data_dir: the directory to store the data.
        device: put the data onto this device.
    Returns:
        train dataset, validation dataset, test dataset.
    """
    dataset = QM9(data_dir, transform=transform)

    # Permute the dataset
    try:
        permu = torch.load("permute.pt")
        dataset = dataset[permu]
    except FileNotFoundError:
        warn("Using non-standard permutation since permute.pt does not exist.")
        dataset, _ = dataset.shuffle(return_perm=True)

    # z score / standard score targets to mean = 0 and std = 1.
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, LABEL_INDEX].item(), std[:, LABEL_INDEX].item()

    # Move the data to the device (it should fit on lisa gpus)
    dataset.data = dataset.data.to(device)

    len_train = 100_000
    len_val = 10_000

    train = dataset[:len_train]
    valid = dataset[len_train : len_train + len_val]
    test = dataset[len_train + len_val :]

    assert len(dataset) == len(train) + len(valid) + len(test)

    return train, valid, test

def get_forward_function(model, model_name, data, Z_ONE_HOT_DIM = 5):
    if model_name == 'TransformerNet':
        data.batch = data.batch[data.ground_node]
        out = model(data.x,
              data.edge_index,
              data.subgraph_edge_index,
              data.node_subnode_index,
              data.subnode_node_index,
              data.ground_node,
              data.subgraph_batch_index,
              data.batch)
        return out
    if model_name == 'FractalNet':
        data.batch = data.batch[data.ground_node]
        out = model(data.x,
              data.edge_index,
              data.subgraph_edge_index,
              data.node_subnode_index,
              data.subnode_node_index,
              data.ground_node,
              data.subgraph_batch_index,
              data.batch)
        return out
    elif model_name == 'FractalNetShared':
        data.batch = data.batch[data.ground_node]
        out = model(data.x[:, :Z_ONE_HOT_DIM],
              data.edge_index,
              data.subgraph_edge_index,
              data.node_subnode_index,
              data.subnode_node_index,
              data.ground_node,
              data.subgraph_batch_index,
              data.batch)
        return out
    elif model_name == 'GNN':
        out = model(data.x[:, :Z_ONE_HOT_DIM],
              data.edge_index,
              data.edge_attr,
              data.batch)
        return out
    elif model_name == 'GNN_no_rel':
        out = model(data.x[:, :Z_ONE_HOT_DIM],
              data.edge_index,
              None,
              data.batch)
        return out
    elif model_name == 'Net':
        out = model(data.x[:, :Z_ONE_HOT_DIM],
              data.edge_index,
              data.batch)
        return out
    else:
        raise ValueError("Model name not recognized")

from tqdm.auto import tqdm

def train_model(model, model_name, epochs, train_loader, valid_loader, test_loader, optimizer, criterion, scheduler, device, LABEL_INDEX=7, Z_ONE_HOT_DIM=5, debug=False, **kwargs):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    train_losses = []
    val_losses = []
    best_val_loss = np.inf

    for epoch in tqdm(range(epochs), desc='Epochs', ncols='100%'):
        if debug:
            out, target = None, None
            for data in train_loader:
                data = data.to(device)
                out = get_forward_function(model, model_name, data, Z_ONE_HOT_DIM)
                target = data.y[:, LABEL_INDEX]
                break
            out, target = out.double(), target.double()
            criterion = criterion.double()
            gradcheck_result = gradcheck(criterion, (out.squeeze(), target), eps=1e-4, atol=1e-4)
            if not gradcheck_result:
                raise ValueError('Gradient check failed')
            else:
                print('Gradient check passed at the beginning of epoch', epoch)

        model.train()
        train_loss = 0

        for data in tqdm(train_loader, desc='Training', ncols='100%'):
            data = data.to(device)
            optimizer.zero_grad()
            target = data.y[:, LABEL_INDEX]
            out = get_forward_function(model, model_name, data, Z_ONE_HOT_DIM)
            loss = criterion(out.squeeze(), target)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for data in tqdm(valid_loader, desc='Validation', ncols='100%'):
                data = data.to(device)
                target = data.y[:, LABEL_INDEX]
                out = get_forward_function(model, model_name, data)
                loss = criterion(out.squeeze(), target)
                valid_loss += loss.item()

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), f'models/{model_name}.pt')

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(valid_loss / len(valid_loader))

        if scheduler is not None:
            scheduler.step(valid_loss / len(valid_loader))

        print(f'Epoch: {epoch}, Loss: {train_loss / len(train_loader)}, Valid Loss: {valid_loss / len(valid_loader)}')

    # Test evaluation
    model.load_state_dict(torch.load(f'models/{model_name}.pt'))
    model.eval()

    test_loss = 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing', ncols='100%'):
            data = data.to(device)
            target = data.y[:, LABEL_INDEX]
            out = get_forward_function(model, model_name, data)
            loss = criterion(out.squeeze(), target)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss}')

    return {'train_loss': train_losses, 'valid_loss': val_losses, 'test_loss': avg_test_loss, 'total_params': total_params}

if __name__ == '__main__':
    # Experiment settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 65
    batch_size = 32
    node_features = 5
    Z_ONE_HOT_DIM = 5
    LABEL_INDEX = 7
    EDGE_ATTR_DIM = 4
    edge_features = 0
    hidden_features = 64
    out_features = 1
    model_name = 'FractalNet'

    # Model, optimizer, and loss function
    model = FractalNet(node_features, edge_features, hidden_features, out_features, depth=4, pool='add', add_residual_skip=False, masking=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)

    # Data preparation
    train, valid, test = get_qm9("data/qm9", device=device, LABEL_INDEX=LABEL_INDEX, transform=Graph_to_Subgraph())
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    # Training and evaluation
    fractalnet_results = train_model(model, model_name, epochs, train_loader, valid_loader, test_loader, optimizer, criterion, scheduler, device, LABEL_INDEX, Z_ONE_HOT_DIM)