import numpy as np
import torch.optim.lr_scheduler
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
import sys
from models.gnn.networks import *
from utils.transforms import Graph_to_Subgraph, Fully_Connected_Graph, Rename_MD17_Features, To_OneHot
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.utils as utils
import torch_geometric.transforms as T
from torch_geometric.datasets import MD17

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

def get_datasets(data_dir, device, name, batch_size, subgraph_dict = None ):
    transforms = [Rename_MD17_Features(), To_OneHot(), Fully_Connected_Graph()]
    if subgraph_dict is not None:
        subgraph_mode = subgraph_dict['mode']
        transforms.append(Graph_to_Subgraph(mode=subgraph_mode))
    transform = T.Compose(transforms)

    dataset = MD17(root=data_dir, name=name, train=True, transform=transform)
    train, valid = dataset[:950], dataset[950:1000]
    test = MD17(root='./data/MD17', name=name, train=False, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def train_md17_model(model, model_name, data_dir, name, subgraph_dict,
                    epochs, batch_size, optimizer, criterion,
                    scheduler, scheduler_name, device, **kwargs):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    if subgraph_dict is not None:
        print("Using subgraph dataset")

    best_val_loss = np.inf
    # write it in the folder logs/md17/name
    writer = SummaryWriter(os.path.join('logs', 'md17', name, model_name))
    writer.add_scalar('Total number of parameters:', total_params)

    train_loader, valid_loader, test_loader = get_datasets(data_dir, device, name, batch_size, subgraph_dict)
    raw_energies = np.array([data.energy.item() for data in train_loader.dataset])
    raw_forces  = np.concatenate([data.force.numpy() for data in train_loader.dataset])

    #print(f"Shift: {shift}, Scale: {scale}")
    shift = np.mean(raw_energies)
    scale = np.sqrt(np.mean(raw_forces **2))

    print(f"Energy shift: {shift}, Energy scale: {scale}")
    print(f"Force shift: {shift}, Force scale: {scale}")
    # Make directories in case they don't exist
    directory = os.path.join('trained', 'md17', name)
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Manual scheduling
    scheduler = CosineWarmupScheduler(optimizer, 50, epochs)
    for epoch in tqdm(range(epochs), desc='Epochs', ncols=100):
        # Training loop
        model.train()
        train_mae_energy, train_mae_force = 0, 0
        training_loss = 0

        for data in tqdm(train_loader, desc='Training', ncols=100, leave=False, position=0, unit='batch', unit_scale=train_loader.batch_size, dynamic_ncols=True, file=sys.stdout):
            data = data.to(device)
            # print all attributes that data has
            data.x = data.x.float()
            data.pos = torch.autograd.Variable(data.pos, requires_grad=True)
            optimizer.zero_grad()
            pred_energy = model(data).squeeze()

            if subgraph_dict is not None:
                #print("is_leaf: ", data.pos[data.ground_node].is_leaf)
                #print("requires_grad: ", data.pos[data.ground_node].requires_grad)
                pred_force_all = -1.0 * torch.autograd.grad(
                    pred_energy,
                    data.pos,
                    grad_outputs=torch.ones_like(pred_energy),
                    create_graph=True,
                    retain_graph=True
                )[0]
                pred_force = pred_force_all[data.ground_node]
                data.force = data.force[data.ground_node]

            else:
                pred_force = -1.0 * torch.autograd.grad(
                    pred_energy,
                    data.pos,
                    grad_outputs=torch.ones_like(pred_energy),
                    create_graph=True,
                    retain_graph=True
                )[0]

            energy_loss = torch.mean((pred_energy - (data.energy-  shift)/scale) ** 2)
            force_loss = torch.mean(torch.sum((pred_force - (data.force)/scale) ** 2, -1)) / 3.

            train_loss = energy_loss + force_loss
            training_loss += train_loss.item()

            mae_energy = criterion(pred_energy * scale + shift, data.energy)
            mae_force = criterion(pred_force * scale + shift, data.force)
            print("energy_loss: ", energy_loss)
            print("force_loss: ", force_loss)
            train_loss.backward()

            '''for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, param.grad.mean(), param.grad.max())
                    # if there is nan in the gradient, then stop
                    if torch.isnan(param.grad).any():
                        print("nan in gradient")
                        sys.exit()'''

            train_mae_energy += mae_energy.item()
            train_mae_force += mae_force.item()

            utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        writer.add_scalar('Training Energy MAE', train_mae_energy / len(train_loader), epoch)
        writer.add_scalar('Training Force MAE', train_mae_force / len(train_loader), epoch)
        writer.add_scalar('Training Loss', training_loss / len(train_loader), epoch)

        # Validation loop
        model.eval()
        val_mae_energy, val_mae_force = 0, 0
        valid_loss = 0
        for data in tqdm(valid_loader, desc='Validation', ncols=100, leave=False, position=0, unit='batch', unit_scale=valid_loader.batch_size, dynamic_ncols=True, file=sys.stdout):
            data = data.to(device)
            data.x = data.x.float()
            data.pos = torch.autograd.Variable(data.pos, requires_grad=True)
            optimizer.zero_grad()

            pred_energy = model(data).squeeze()
            if subgraph_dict is not None:
                pred_force_all = -1.0 * torch.autograd.grad(
                    pred_energy,
                    data.pos,
                    grad_outputs=torch.ones_like(pred_energy),
                    create_graph=True,
                    retain_graph=True
                )[0]
                pred_force = pred_force_all[data.ground_node]
                data.force = data.force[data.ground_node]

            else:
                pred_force = -1.0 * torch.autograd.grad(
                    pred_energy,
                    data.pos,
                    grad_outputs=torch.ones_like(pred_energy),
                    create_graph=True,
                    retain_graph=True
                )[0]
            energy_loss = torch.mean((pred_energy - (data.energy-  shift)/scale) ** 2)
            force_loss = torch.mean(torch.sum((pred_force - (data.force - shift)/scale) ** 2, -1)) / 3.

            mae_energy = criterion(pred_energy * scale + shift, data.energy)
            mae_force = criterion(pred_force * scale + shift, data.force)
            #mae_energy = criterion(pred_energy * scale + shift, data.energy)
            #mae_force = criterion(pred_force * scale, data.force)

            #energy_loss = torch.mean((pred_energy - (data.energy - shift) / scale) ** 2)
            #force_loss = torch.mean(torch.sum((pred_force - data.force / scale) ** 2, -1)) / 3.
            total_val_loss = energy_loss + force_loss
            valid_loss += total_val_loss.item()

            val_mae_energy += mae_energy.item()
            val_mae_force += mae_force.item()

        writer.add_scalar('Validation Energy MAE', val_mae_energy / len(valid_loader), epoch)
        writer.add_scalar('Validation Force MAE', val_mae_force / len(valid_loader), epoch)
        writer.add_scalar('Validation Loss', valid_loss / len(valid_loader), epoch)
        # Save model if validation loss is lower than previous best
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), f'trained/md17/{name}/{model_name}.pt')

        if scheduler is not None:
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(valid_loss / len(valid_loader))
            if scheduler_name == "CosineAnnealingLR":
                scheduler.step()
            else:
                scheduler.step()
        # log the learning rate after the scheduler
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        print(f'Epoch: {epoch}, Loss: {training_loss / len(train_loader)}, Valid Loss: {valid_loss / len(valid_loader)}', end='\r')

    # Test evaluation
    model.load_state_dict(torch.load(f'trained/md17/{name}/{model_name}.pt'))
    model.eval()

    test_mae_energy, test_mae_force = 0, 0
    test_loss = 0

    for data in tqdm(test_loader, desc='Testing', ncols=100, leave=False, position=0, unit='batch', unit_scale=test_loader.batch_size, dynamic_ncols=True, file=sys.stdout):
        data = data.to(device)
        data.x = data.x.float()
        data.pos = torch.autograd.Variable(data.pos, requires_grad=True)
        optimizer.zero_grad()

        pred_energy = model(data).squeeze()
        if subgraph_dict is not None:
            pred_force_all = -1.0 * torch.autograd.grad(
                pred_energy,
                data.pos,
                grad_outputs=torch.ones_like(pred_energy),
                create_graph=True,
                retain_graph=True
            )[0]
            pred_force = pred_force_all[data.ground_node]
            data.force = data.force[data.ground_node]

        else:
            pred_force = -1.0 * torch.autograd.grad(
                pred_energy,
                data.pos,
                grad_outputs=torch.ones_like(pred_energy),
                create_graph=True,
                retain_graph=True
            )[0]
        energy_loss = torch.mean((pred_energy - (data.energy-  shift)/scale) ** 2)
        force_loss = torch.mean(torch.sum((pred_force - (data.force - shift)/scale) ** 2, -1)) / 3.

        mae_energy = criterion(pred_energy * scale + shift, data.energy)
        mae_force = criterion(pred_force * scale + shift, data.force)
        #mae_energy = criterion(pred_energy * scale + shift, data.energy)
        #mae_force = criterion(pred_force * scale, data.force)

        #energy_loss = torch.mean((pred_energy - (data.energy - shift) / scale) ** 2)
        #force_loss = torch.mean(torch.sum((pred_force - data.force / scale) ** 2, -1)) / 3.
        total_test_loss = energy_loss + force_loss
        test_loss += total_test_loss.item()

        test_mae_energy += mae_energy.item()
        test_mae_force += mae_force.item()
    writer.add_scalar('Test Energy MAE', test_mae_energy / len(test_loader), epoch)
    writer.add_scalar('Test Force MAE', test_mae_force / len(test_loader), epoch)
    writer.add_scalar('Test Loss', test_loss / len(test_loader), epoch)
    print(f'Test Energy MAE: {test_mae_energy / len(test_loader)}')
    print(f'Test Force MAE: {test_mae_force / len(test_loader)}')
    print(f'Test Loss: {test_loss / len(test_loader)}')

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
    criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)

    # Data preparation
    train, valid, test = get_qm9("./data/qm9", device=device, LABEL_INDEX=LABEL_INDEX, transform=Graph_to_Subgraph())
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    # Training and evaluation
    fractalnet_results = train_qm9_model(model, epochs, train_loader, valid_loader, test_loader, optimizer, criterion, scheduler, device, LABEL_INDEX, Z_ONE_HOT_DIM)