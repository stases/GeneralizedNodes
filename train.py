import torch
import tqdm.notebook as tqdm
import numpy as np
from warnings import warn
from torch_geometric.datasets import QM9
import time

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

    if model_name == 'FractalNet':
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

def train_model(model, model_name, epochs, train_loader, valid_loader, test_loader, optimizer, criterion, scheduler, device, LABEL_INDEX = 7, Z_ONE_HOT_DIM = 5, **kwargs):
    """Train the model for a number of epochs.
    Args:
        model: the model to train.
        epochs: the number of epochs to train for.
        train_loader: the data loader for the training data.
        optimizer: the optimizer to use.
        criterion: the loss function to use.
        device: the device to train on.
    Returns:
        #TODO: what does it return
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    for epoch in tqdm.tqdm(range(epochs)):
        model.train()
        train_loss = 0
        for data in tqdm.tqdm(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                target = data.y[:, LABEL_INDEX]
                # keep only ground nodes in the data.batch
                # check if there is ground_node in data.batch
                out = get_forward_function(model, model_name, data, Z_ONE_HOT_DIM)
                loss = criterion(out.squeeze(), target)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                # show loss on tqdm
                #tqdm.tqdm.write(f'Epoch: {epoch}, Loss: {loss.item()}')
            # store loss per epoch
            # get performance on the validation set
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for data in tqdm.tqdm(valid_loader):
                data = data.to(device)
                target = data.y[:, LABEL_INDEX]
                out = get_forward_function(model, model_name, data)
                loss = criterion(out.squeeze(), target)
                valid_loss += loss.item()
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            # save the model with the best validation loss and have it name as the model name in the models folder
            torch.save(model.state_dict(), f'models/{model_name}.pt')
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(valid_loss/len(valid_loader))
        if scheduler is not None:
            scheduler.step(valid_loss / len(valid_loader))
        print(f'Epoch: {epoch}, Loss: {train_loss/len(train_loader)},'
              f' Valid Loss: {valid_loss/len(valid_loader)}')
    # get performance on the test set
    model.load_state_dict(torch.load(f'models/{model_name}.pt'))
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for data in tqdm.tqdm(test_loader):
            data = data.to(device)
            target = data.y[:, LABEL_INDEX]
            out = get_forward_function(model, model_name, data)
            loss = criterion(out.squeeze(), target)
            test_loss += loss.item()
    avg_test_loss = test_loss/len(test_loader)
    print(f'Test Loss: {test_loss/len(test_loader)}')
    return {'train_loss': train_losses, 'valid_loss': val_losses, 'test_loss': avg_test_loss ,'total_params': total_params}