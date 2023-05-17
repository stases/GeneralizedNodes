import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
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
#from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

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

class Task(pl.LightningModule):
    """Graph Neural Network module"""

    def __init__(
        self,
        model,
        lr,
        weight_decay,
        warmup_epochs,
        weight=1,
        shift=0,
        scale=1,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.weight = weight
        self.shift = shift
        self.scale = scale

        self.energy_train_metric = torchmetrics.MeanAbsoluteError()
        self.energy_valid_metric = torchmetrics.MeanAbsoluteError()
        self.energy_test_metric = torchmetrics.MeanAbsoluteError()
        self.force_train_metric = torchmetrics.MeanAbsoluteError()
        self.force_valid_metric = torchmetrics.MeanAbsoluteError()
        self.force_test_metric = torchmetrics.MeanAbsoluteError()

    def forward(self, graph):
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
        return pred_energy.squeeze(-1), pred_force

    def energy_and_force_loss(self, graph, energy, force):
        loss = F.mse_loss(energy, (graph.energy - self.shift) / self.scale)
        loss += self.weight * F.mse_loss(force, graph.force / self.scale)
        return loss

    def training_step(self, graph):
        energy, force = self(graph)

        # print("train", energy * self.scale + self.shift - graph.energy)

        loss = self.energy_and_force_loss(graph, energy, force)
        self.energy_train_metric(energy * self.scale + self.shift, graph.energy)
        self.force_train_metric(force * self.scale, graph.force)

        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        return loss

    @torch.inference_mode(False)
    def validation_step(self, graph, batch_idx):
        energy, force = self(graph)
        # print("valid", energy * self.scale + self.shift - graph.energy)
        self.energy_valid_metric(energy * self.scale + self.shift, graph.energy)
        self.force_valid_metric(force * self.scale, graph.force)


    @torch.inference_mode(False)
    def test_step(self, graph, batch_idx):
        energy, force = self(graph)
        self.energy_test_metric(energy * self.scale + self.shift, graph.energy)
        self.force_test_metric(force * self.scale, graph.force)


    def train_dataloader(self):
        train_loader, _, _ = get_datasets(self.data_dir, self.device, self.name, self.batch_size, self.subgraph_dict)
        return train_loader

    def val_dataloader(self):
        _, valid_loader, _ = get_datasets(self.data_dir, self.device, self.name, self.batch_size, self.subgraph_dict)
        return valid_loader

    def test_dataloader(self):
        _, _, test_loader = get_datasets(self.data_dir, self.device, self.name, self.batch_size, self.subgraph_dict)
        return test_loader

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        # )
        optimizer = torch.optim.RAdam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        num_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = num_steps / self.trainer.max_epochs

        if self.warmup_epochs == 0:
            self.warmup_epochs = 1 / steps_per_epoch

        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=self.warmup_epochs * steps_per_epoch,
        #     max_epochs=num_steps,
        # )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(
    #         self.parameters(), lr=self.lr, weight_decay=self.weight_decay
    #     )
    #
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, self.trainer.max_epochs, verbose=False
    #     )
    #     lr_scheduler_config = {
    #         "scheduler": scheduler,
    #         "interval": "epoch",
    #         "frequency": 1,
    #     }
    #     return [optimizer], [lr_scheduler_config]