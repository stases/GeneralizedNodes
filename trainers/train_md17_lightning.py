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
import torchmetrics

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
    test = MD17(root=data_dir, name=name, train=False, transform=transform)

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
    def __init__(self, model,data_dir, name, batch_size, subgraph_dict=None, **kwargs):
        super().__init__()
        self.model = model
        self.data_dir = data_dir
        self.name = name
        self.subgraph_dict = subgraph_dict
        self.batch_size = batch_size
        self.weight = 1.0
        self.learning_rate = kwargs['learning_rate']
        self.energy_train_metric = torchmetrics.MeanAbsoluteError()
        self.energy_valid_metric = torchmetrics.MeanAbsoluteError()
        self.energy_test_metric = torchmetrics.MeanAbsoluteError()
        self.force_train_metric = torchmetrics.MeanAbsoluteError()
        self.force_valid_metric = torchmetrics.MeanAbsoluteError()
        self.force_test_metric = torchmetrics.MeanAbsoluteError()
        self.shift, self.scale = get_shift_scale(self.train_dataloader())
        self.save_hyperparameters(ignore=['criterion', 'model'])

    def forward(self, graph):
        graph = graph.to(self.device)
        graph.x = graph.x.float()
        energy, force = self.pred_energy_and_force(graph)
        return energy, force

    def pred_energy_and_force(self, graph):
        graph.pos = torch.autograd.Variable(graph.pos, requires_grad=True)
        pred_energy = self.model(graph)
        sign = -1.0
        pred_force = (
                sign
                * torch.autograd.grad(
            pred_energy,
            graph.pos,
            grad_outputs=torch.ones_like(pred_energy),
            create_graph=True,
            retain_graph=True,
        )[0]
        )
        if self.subgraph_dict is not None:
            pred_force = pred_force[graph.ground_node]
            graph.force = graph.force[graph.ground_node]

        return pred_energy.squeeze(-1), pred_force

    def energy_and_force_loss(self, graph, energy, force):
        loss = F.mse_loss(energy, (graph.energy - self.shift) / self.scale)
        #print("Energy loss", loss.item())
        loss += self.weight * F.mse_loss(force, graph.force / self.scale)
        #print("Force loss", loss.item())
        return loss

    def training_step(self, batch, batch_idx):
        graph = batch.to(self.device)
        graph.x = graph.x.float()
        energy, force = self.pred_energy_and_force(graph)

        loss = self.energy_and_force_loss(graph, energy, force)
        self.energy_train_metric(energy * self.scale + self.shift, graph.energy)
        self.force_train_metric(force * self.scale, graph.force)

        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_step=True)

        return loss

    def on_train_epoch_end(self):
        self.log("Energy train MAE", self.energy_train_metric, prog_bar=True)
        self.log("Force train MAE", self.force_train_metric, prog_bar=True)
        print("Current learning rate: ", self.trainer.optimizers[0].param_groups[0]["lr"])

    @torch.inference_mode(False)
    def validation_step(self, graph, batch_idx):
        energy, force = self(graph)
        # print("valid", energy * self.scale + self.shift - graph.energy)
        self.energy_valid_metric(energy * self.scale + self.shift, graph.energy)
        self.force_valid_metric(force * self.scale, graph.force)

    def on_validation_epoch_end(self):
        self.log("Energy valid MAE", self.energy_valid_metric, prog_bar=True)
        self.log("Force valid MAE", self.force_valid_metric, prog_bar=True)

    @torch.inference_mode(False)
    def test_step(self, graph, batch_idx):
        energy, force = self(graph)
        self.energy_test_metric(energy * self.scale + self.shift, graph.energy)
        self.force_test_metric(force * self.scale, graph.force)

    def on_test_epoch_end(self):
        self.log("Energy test MAE", self.energy_test_metric, prog_bar=True)
        self.log("Force test MAE", self.force_test_metric, prog_bar=True)
        # log the number of parameters of the model
        self.log("Number of parameters", sum(p.numel() for p in self.parameters() if p.requires_grad), prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        warmup_epochs = np.ceil(self.trainer.max_epochs * 0.05)
        scheduler = CosineWarmupScheduler(optimizer, warmup=warmup_epochs, max_iters=self.trainer.max_epochs)
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