import numpy as np
import torch
import torch.nn as nn
from torch.autograd import gradcheck
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
import sys
from .get_fw_function import get_forward_function
from .get_qm9 import get_qm9, rescale, get_mean_std, get_qm9_statistics
from models.gnn.networks import FractalNet, FractalNetShared, GNN, GNN_no_rel, Net, TransformerNet
from utils.subgraph import Graph_to_Subgraph
from torch.utils.tensorboard import SummaryWriter
import os

def path_finder(dir, file):
    parent_dir = dir
    name = file
    path = os.path.join(parent_dir, name)

    if os.path.exists(path):
        i = 1
        while os.path.exists(path + '_' + str(i)):
            i += 1
        name = name + '_' + str(i)
        path = os.path.join(parent_dir, name)
    return path

def get_datasets(data_dir, device, LABEL_INDEX, subgraph, batch_size):
    transform = Graph_to_Subgraph() if subgraph else None
    if subgraph:
        print("Using subgraph dataset.\n")
    else:
        print("Using regular dataset.\n")
    train, valid, test = get_qm9(data_dir, device=device, LABEL_INDEX=LABEL_INDEX, transform=Graph_to_Subgraph())
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader
def train_qm9_model(model, model_name, data_dir, subgraph, epochs, batch_size, optimizer, criterion, scheduler, device, LABEL_INDEX=7, Z_ONE_HOT_DIM=5, debug=False, **kwargs):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    train_losses = []
    val_losses = []

    best_val_loss = np.inf
    #writer = SummaryWriter(path_finder('logs', model_name))
    writer = SummaryWriter('logs/' + model_name)
    train_loader, valid_loader, test_loader = get_datasets(data_dir, device, LABEL_INDEX, subgraph, batch_size)
    for epoch in tqdm(range(epochs), desc='Epochs', ncols=100):
        if debug:
            out, target = None, None
            for data in train_loader:
                data = data.to(device)
                out = get_forward_function(model, data, Z_ONE_HOT_DIM)
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

        for data in tqdm(train_loader, desc='Training', ncols=100, leave=False, position=0, unit='batch', unit_scale=train_loader.batch_size, dynamic_ncols=True, file=sys.stdout):
            data = data.to(device)
            optimizer.zero_grad()
            target = data.y[:, LABEL_INDEX]
            out = get_forward_function(model, data, Z_ONE_HOT_DIM)
            loss = criterion(out.squeeze(), target)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        writer.add_scalar('Training Loss', train_loss / len(train_loader), epoch)
        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for data in tqdm(valid_loader, desc='Validation', ncols=100, leave=False, position=0, unit='batch', unit_scale=valid_loader.batch_size, dynamic_ncols=True, file=sys.stdout):
                data = data.to(device)
                target = data.y[:, LABEL_INDEX]
                out = get_forward_function(model, data, Z_ONE_HOT_DIM)
                loss = criterion(out.squeeze(), target)
                valid_loss += loss.item()
        writer.add_scalar('Validation Loss', valid_loss / len(valid_loader), epoch)
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), f'trained/qm9/{model_name}.pt')

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(valid_loss / len(valid_loader))

        if scheduler is not None:
            scheduler.step(valid_loss / len(valid_loader))

        print(f'Epoch: {epoch}, Loss: {train_loss / len(train_loader)}, Valid Loss: {valid_loss / len(valid_loader)}', end='\r')

    # Test evaluation
    model.load_state_dict(torch.load(f'models/{model_name}.pt'))
    model.eval()

    test_loss = 0
    unnormalized_loss = 0
    with torch.no_grad():
        # Get the dataset mean and std from the get_qm9.py file
        mean, std = get_qm9_statistics('.data/qm9')
        mean, std = mean[:, LABEL_INDEX], std[:, LABEL_INDEX]
        for data in tqdm(test_loader, desc='Testing', ncols=100, leave=False, position=0, unit='batch', unit_scale=test_loader.batch_size, dynamic_ncols=True, file=sys.stdout):
            data = data.to(device)
            target = data.y[:, LABEL_INDEX]
            out = get_forward_function(model, data, Z_ONE_HOT_DIM)
            loss = criterion(out.squeeze(), target)
            unnormalized_output, unnormalized_target = rescale(out.squeeze(), mean, std), rescale(target, mean, std)
            unnormalized_loss += criterion(unnormalized_output, unnormalized_target).item()
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss}')
    avg_unnormalized_loss = unnormalized_loss / len(test_loader)
    writer.add_scalar('Test Loss', avg_test_loss)
    writer.add_scalar('Rescaled Test Loss', avg_unnormalized_loss)
    writer.close()
    return {'train_loss': train_losses, 'valid_loss': val_losses, 'test_loss': avg_test_loss, 'rescaled_test_loss': avg_unnormalized_loss}

if __name__ == '__main__':
    # Experiment settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 1
    batch_size = 32
    node_features = 5
    Z_ONE_HOT_DIM = 5
    LABEL_INDEX = 7
    EDGE_ATTR_DIM = 4
    edge_features = 0
    hidden_features = 64
    out_features = 1

    # Model, optimizer, and loss function
    model = FractalNet(node_features, edge_features, hidden_features, out_features, depth=4, pool='add', add_residual_skip=False, masking=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)

    # Data preparation
    train, valid, test = get_qm9("./data/qm9", device=device, LABEL_INDEX=LABEL_INDEX, transform=Graph_to_Subgraph())
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    # Training and evaluation
    fractalnet_results = train_qm9_model(model, epochs, train_loader, valid_loader, test_loader, optimizer, criterion, scheduler, device, LABEL_INDEX, Z_ONE_HOT_DIM)