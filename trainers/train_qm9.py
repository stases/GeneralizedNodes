import numpy as np
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
import sys
from models.gnn.networks import *
from utils.transforms import Graph_to_Subgraph, Fully_Connected_Graph
from utils.tools import compute_mean_mad
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.utils as utils
from torch_geometric.datasets import QM9


def get_qm9(data_dir, device="cuda", LABEL_INDEX = 7, transform=None):
    dataset = QM9(data_dir, transform=transform)
    dataset.data = dataset.data.to(device)

    len_train = 100_000
    len_test = 10_000

    train = dataset[:len_train]
    valid = dataset[len_train + len_test:]
    test = dataset[len_train : len_train + len_test]
    assert len(dataset) == len(train) + len(valid) + len(test)

    return train, valid, test

def get_datasets(data_dir, device, LABEL_INDEX, subgraph, batch_size, fully_connect=False):
    if subgraph:
        transform = Graph_to_Subgraph(fully_connect=fully_connect)
    elif fully_connect and not subgraph:
        transform = Fully_Connected_Graph()
    else:
        transform = None

    train, valid, test = get_qm9(data_dir, device=device, LABEL_INDEX=LABEL_INDEX, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

def train_qm9_model(model, model_name, data_dir, subgraph, fully_connect,
                    epochs, batch_size, optimizer, criterion,
                    scheduler, scheduler_name, device,
                    LABEL_INDEX=7, Z_ONE_HOT_DIM=5,
                    debug=False, **kwargs):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    print("Using subgraph dataset: ", subgraph)
    print("Using fully connected subgraph dataset: ", fully_connect)

    train_losses = []
    val_losses = []
    best_val_loss = np.inf

    writer = SummaryWriter('logs/' + model_name)
    writer.add_scalar('Total number of parameters:', total_params)

    train_loader, valid_loader, test_loader = get_datasets(data_dir, device, LABEL_INDEX, subgraph, batch_size, fully_connect)
    mean, mad = compute_mean_mad(train_loader, LABEL_INDEX)

    for epoch in tqdm(range(epochs), desc='Epochs', ncols=100):
        # Training loop
        model.train()
        train_loss, train_mae = 0, 0
        for data in tqdm(train_loader, desc='Training', ncols=100, leave=False, position=0, unit='batch', unit_scale=train_loader.batch_size, dynamic_ncols=True, file=sys.stdout):
            data = data.to(device)
            optimizer.zero_grad()
            target = data.y[:, LABEL_INDEX]
            out = model(data).squeeze()
            loss = criterion(out, (target - mean) / mad)
            mae = criterion(out * mad + mean, target)
            loss.backward()
            train_loss += loss.item()
            train_mae += mae.item()
            utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        writer.add_scalar('Training Loss', train_loss / len(train_loader), epoch)
        writer.add_scalar('Training MAE', train_mae / len(train_loader), epoch)

        # Validation loop
        model.eval()
        valid_loss, valid_mae = 0, 0
        with torch.no_grad():
            for data in tqdm(valid_loader, desc='Validation', ncols=100, leave=False, position=0, unit='batch', unit_scale=valid_loader.batch_size, dynamic_ncols=True, file=sys.stdout):
                data = data.to(device)
                target = data.y[:, LABEL_INDEX]
                out = model(data).squeeze()
                loss = criterion(out, (target - mean) / mad)
                mae = criterion(out * mad + mean, target)
                valid_loss += loss.item()
                valid_mae += mae.item()
        writer.add_scalar('Validation Loss', valid_loss / len(valid_loader), epoch)
        writer.add_scalar('Validation MAE', valid_mae / len(valid_loader), epoch)

        # Save model if validation loss is lower than previous best
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), f'trained/qm9/{model_name}.pt')

        if scheduler is not None:
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(valid_loss / len(valid_loader))
            if scheduler_name == "CosineAnnealingLR":
                scheduler.step()
        # log the learning rate after the scheduler
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        print(f'Epoch: {epoch}, Loss: {train_loss / len(train_loader)}, Valid Loss: {valid_loss / len(valid_loader)}', end='\r')

    # Test evaluation
    model.load_state_dict(torch.load(f'trained/qm9/{model_name}.pt'))
    model.eval()

    test_loss, test_mae = 0, 0
    unnormalized_loss = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing', ncols=100, leave=False, position=0, unit='batch', unit_scale=test_loader.batch_size, dynamic_ncols=True, file=sys.stdout):
            data = data.to(device)
            target = data.y[:, LABEL_INDEX]
            out = model(data).squeeze()
            loss = criterion(out, (target - mean) / mad)
            mae = criterion(out * mad + mean, target)
            test_loss += loss.item()
            test_mae += mae.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss}')
    writer.add_scalar('Test Loss', avg_test_loss)
    writer.add_scalar('Test MAE', test_mae / len(test_loader))
    writer.close()

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
    model = FractalNet(node_features, edge_features, hidden_features, out_features, depth=4, pool='add', residual=False, masking=True).to(device)
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