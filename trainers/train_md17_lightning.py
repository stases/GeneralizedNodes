import numpy as np
import torch.optim.lr_scheduler
from torch_geometric.loader import DataLoader
from models.gnn.networks import *
from utils.transforms import Graph_to_Subgraph, Fully_Connected_Graph, Rename_MD17_Features, To_OneHot
import torch_geometric.transforms as T
from torch_geometric.datasets import MD17
import torch
import pytorch_lightning as pl
import wandb

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

def get_shift_scale(train_loader):
    raw_energies = np.array([data.energy.item() for data in train_loader.dataset])
    raw_forces  = np.concatenate([data.force.numpy() for data in train_loader.dataset])
    energy_shift = np.mean(raw_energies)
    force_shift = np.mean(raw_forces, axis=0)
    force_scale = np.sqrt(np.mean((raw_forces) **2))
    shift = energy_shift
    scale = force_scale
    return shift, scale

class MD17Model(pl.LightningModule):
    def __init__(self, model, model_name, data_dir, name, subgraph_dict, batch_size, criterion, **kwargs):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.data_dir = data_dir
        self.name = name
        self.subgraph_dict = subgraph_dict
        self.batch_size = batch_size
        self.criterion = criterion
        self.save_hyperparameters()
        self.shift, self.scale = get_shift_scale(self.train_dataloader())
        
    def forward(self, data):
        return self.model(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=1000)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_loader, _, _ = get_datasets(self.data_dir, self.device, self.name, self.batch_size, self.subgraph_dict)
        return train_loader

    def val_dataloader(self):
        _, valid_loader, _ = get_datasets(self.data_dir, self.device, self.name, self.batch_size, self.subgraph_dict)
        return valid_loader

    def test_dataloader(self):
        _, _, test_loader = get_datasets(self.data_dir, self.device, self.name, self.batch_size, self.subgraph_dict)
        return test_loader

    def training_step(self, batch, batch_idx):
        data = batch.to(self.device)
        data.x = data.x.float()
        data.pos = torch.autograd.Variable(data.pos, requires_grad=True)
        pred_energy = self.model(data).squeeze()
        if self.subgraph_dict is not None:
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

        energy_loss = torch.mean((pred_energy - (data.energy - self.shift) / self.scale) ** 2)
        force_loss = torch.mean(torch.sum((pred_force - (data.force ) / self.scale) ** 2, -1)) / 3.

        train_loss = energy_loss + force_loss
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):

        '''
        data = batch.to(self.device)
        data.x = data.x.float()
        data.pos = torch.autograd.Variable(data.pos, requires_grad=True)
        with torch.enable_grad():
            pred_energy = self.model(data).squeeze()

            if self.subgraph_dict is not None:
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

        energy_loss = torch.mean((pred_energy - (data.energy - self.shift) / self.scale) ** 2)
        force_loss = torch.mean(torch.sum((pred_force - (data.force) / self.scale) ** 2, -1)) / 3.

        self.log('val_loss', energy_loss + force_loss)'''
        with torch.enable_grad():
            self.train()
            return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        with torch.enable_grad():
            self.train()
            return self.training_step(batch, batch_idx)

def main():
    # Instantiate your model and set up the necessary parameters
    model = MD17Model(...)
    # Set up your data loaders and other necessary components

    # Initialize the Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        gpus=1,
        progress_bar_refresh_rate=20,
    )

    # Start the training
    trainer.fit(model)

    # Evaluate on the test set
    trainer.test(model)

    # Save the trained model
    torch.save(model.state_dict(), 'path/to/save/model.pt')


if __name__ == '__main__':
    main()