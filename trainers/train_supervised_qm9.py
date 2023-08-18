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
from torch_geometric.datasets import QM9
from utils.supervised_qm9.QM9_hypernode import QM9_Hypernodes

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

def compute_mean_mad(train_loader, label_property):
    values = train_loader.dataset.data.y[:, label_property]
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad

def get_qm9_hypernodes(data_dir, mode, transform=None, device="cuda"):
    dataset = QM9_Hypernodes(data_dir, mode=mode)
    dataset.data = dataset.data.to(device)

    len_train = 100_000
    len_test = 10_000

    train = dataset[:len_train]
    valid = dataset[len_train + len_test:]
    test = dataset[len_train : len_train + len_test]
    assert len(dataset) == len(train) + len(valid) + len(test)


    return train, valid, test

def get_datasets(data_dir, device, LABEL_INDEX, batch_size, fully_connect=False, subgraph_dict=None):
    transforms = []
    #TODO: Do not make this hardcoded
    mode = "en_ee_nn"
    if fully_connect:
        transforms.append(Fully_Connected_Graph())
    if subgraph_dict is not None:
        subgraph_mode = subgraph_dict.get("mode", None)
        transforms.append(Graph_to_Subgraph(mode=subgraph_mode))
    if len(transforms) > 0:
        transform = T.Compose(transforms)
    else:
        transform = None
    train, valid, test = get_qm9_hypernodes(data_dir, mode, device=device, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=16)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=16)
    return train_loader, valid_loader, test_loader



class SupervisedQM9Model(pl.LightningModule):
    def __init__(self, model, data_dir, LABEL_INDEX, batch_size, warmup_epochs, fully_connect, subgraph_dict=None, **kwargs):
        super().__init__()
        self.model = model
        self.data_dir = data_dir
        self.LABEL_INDEX = LABEL_INDEX
        self.fully_connect = fully_connect
        self.subgraph_dict = subgraph_dict
        self.batch_size = batch_size
        self.criterion = torch.nn.L1Loss()
        self.warmup_epochs = warmup_epochs
        self.learning_rate = kwargs['learning_rate']
        self.energy_train_metric = torchmetrics.MeanAbsoluteError()
        self.energy_valid_metric = torchmetrics.MeanAbsoluteError()
        self.energy_test_metric = torchmetrics.MeanAbsoluteError()
        self.mean, self.mad = compute_mean_mad(self.train_dataloader(), self.LABEL_INDEX)
        self.save_hyperparameters(ignore=['criterion', 'model'])


    def training_step(self, batch, batch_idx):
        graph = batch.to(self.device)
        graph.x = graph.x.float()

        target = batch.y[:, self.LABEL_INDEX]
        out = self.model(graph).squeeze()
        loss = self.criterion(out, (target-self.mean)/self.mad)
        self.energy_train_metric(out*self.mad+self.mean, target)
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_step=True)

        return loss

    def on_train_epoch_end(self):
        self.log("Train MAE", self.energy_train_metric, prog_bar=True)
        print("Current learning rate: ", self.trainer.optimizers[0].param_groups[0]["lr"])

    def validation_step(self, batch, batch_idx):
        graph = batch.to(self.device)
        graph.x = graph.x.float()
        target = batch.y[:, self.LABEL_INDEX]
        out = self.model(graph).squeeze()
        loss = self.criterion(out, (target-self.mean)/self.mad)
        self.val_loss = loss.item()
        self.energy_valid_metric(out*self.mad+self.mean, target)
        return loss

    def on_validation_epoch_end(self):
        self.log("Energy valid MAE", self.energy_valid_metric, prog_bar=True)
        self.log("val_loss", self.val_loss, prog_bar=True)


    def test_step(self, batch, batch_idx):
        graph = batch.to(self.device)
        graph.x = graph.x.float()
        target = batch.y[:, self.LABEL_INDEX]
        out = self.model(graph).squeeze()
        self.energy_test_metric(out * self.mad + self.mean, target)

    def on_test_epoch_end(self):
        self.log("Energy test MAE", self.energy_test_metric, prog_bar=True)
        # log the number of parameters of the model
        self.log("Number of parameters", sum(p.numel() for p in self.parameters() if p.requires_grad), prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        warmup_epochs = self.warmup_epochs
        scheduler = CosineWarmupScheduler(optimizer, warmup=warmup_epochs, max_iters=self.trainer.max_epochs)
        return [optimizer], [scheduler]

    #data_dir, device, LABEL_INDEX, batch_size, fully_connect=False, subgraph_dict=None
    def train_dataloader(self):
        train_loader, _, _ = get_datasets(self.data_dir, self.device, self.LABEL_INDEX, self.batch_size, self.fully_connect, self.subgraph_dict)
        return train_loader

    def val_dataloader(self):
        _, valid_loader, _ = get_datasets(self.data_dir, self.device, self.LABEL_INDEX, self.batch_size, self.fully_connect, self.subgraph_dict)
        return valid_loader

    def test_dataloader(self):
        _, _, test_loader = get_datasets(self.data_dir, self.device, self.LABEL_INDEX, self.batch_size, self.fully_connect, self.subgraph_dict)
        return test_loader