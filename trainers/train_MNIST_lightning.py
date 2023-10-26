import numpy as np
from utils.transforms import Graph_to_Subgraph, Fully_Connected_Graph, Rename_MD17_Features, To_OneHot
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
from torch_geometric.transforms import RadiusGraph, Compose, BaseTransform, Distance, Cartesian, RandomRotate
import torch_geometric as tg

#pl.seed_everything(42, workers=True)

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


def get_datasets(data_dir, batch_size, radius, subgraph_dict=None):
    transforms = []
    transforms.append(RadiusGraph(radius))
    if subgraph_dict is not None:
        subgraph_mode = subgraph_dict.get("mode", None)
        transforms.append(Graph_to_Subgraph(mode=subgraph_mode))
    transforms = Compose(transforms)
    train_val_set = tg.datasets.MNISTSuperpixels(root=data_dir, transform=transforms, train=True)
    # split train into train and val sets by taking the last 10% of the training set
    train_set = train_val_set[:int(len(train_val_set) * 0.9)]
    # make the train set size of one batch
    #train_set = train_set[:batch_size]
    # take only 50% of the train set to be the train set
    #train_set = train_set[:int(len(train_set) * 0.1)]
    #TODO: Remove the line above later
    val_set = train_val_set[int(len(train_val_set) * 0.9):]
    test_set = tg.datasets.MNISTSuperpixels(root=data_dir, transform=transforms, train=False)
    # print which transforms are we using
    print("Transforms: ", transforms)
    #assert len(train_set) + len(val_set) == len(train_val_set)

    train_loader = tg.loader.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = tg.loader.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = tg.loader.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

class MNISTModel(pl.LightningModule):
    def __init__(self, model, data_dir, radius, batch_size, warmup_epochs, subgraph_dict=None, **kwargs):
        super().__init__()
        self.model = model
        print(self.model)
        self.data_dir = data_dir
        self.radius = radius
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.subgraph_dict = subgraph_dict
        self.learning_rate = kwargs["learning_rate"]

        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, graph):
        return self.model(graph)

    def training_step(self, graph):
        graph = graph.to(self.device)
        pred = self(graph).squeeze()
        #print("pred is: ", pred)
        #print("pred shape is: ", pred.shape)
        loss = F.cross_entropy(pred, graph.y.long())

        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.train_metric(pred, graph.y)
        self.train_loss = loss.item()
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train acc", self.train_metric, prog_bar=True)
        self.log("train loss", self.train_loss, prog_bar=True)

    def validation_step(self, graph, batch_idx):
        graph = graph.to(self.device)
        pred = self(graph).squeeze()

        loss = F.cross_entropy(pred, graph.y.long())
        self.val_loss = loss.item()
        self.valid_metric(pred, graph.y)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss, prog_bar=True)
        self.log("valid acc", self.valid_metric, prog_bar=True)

    def test_step(self, graph, batch_idx):
        graph = graph.to(self.device)
        pred = self(graph).squeeze()
        loss = F.cross_entropy(pred,  graph.y.long())
        self.test_metric(pred, graph.y)
        return loss

    def on_test_epoch_end(self):
        self.log("test acc", self.test_metric, prog_bar=True)
        self.log("Number of parameters", sum(p.numel() for p in self.parameters() if p.requires_grad), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        warmup_epochs = self.warmup_epochs
        scheduler = CosineWarmupScheduler(optimizer, warmup=warmup_epochs, max_iters=self.trainer.max_epochs)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_loader, _, _ = get_datasets(self.data_dir, self.batch_size, self.radius, self.subgraph_dict)
        return train_loader

    def val_dataloader(self):
        _, val_loader, _ = get_datasets(self.data_dir, self.batch_size, self.radius, self.subgraph_dict)
        return val_loader

    def test_dataloader(self):
        _, _, test_loader = get_datasets(self.data_dir, self.batch_size, self.radius, self.subgraph_dict)
        return test_loader
