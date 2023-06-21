import numpy as np
from utils.transforms import Graph_to_Subgraph, Fully_Connected_Graph, Rename_MD17_Features, To_OneHot
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
from torch_geometric.transforms import RadiusGraph, Compose, BaseTransform, Distance, Cartesian, RandomRotate
import torch_geometric as tg
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import os
from typing import Callable, List, Optional
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from tqdm import tqdm
import os
from typing import Callable, List, Optional
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from tqdm import tqdm
from geomloss import SamplesLoss

def sinkhorn_loss(x, y):
    # "sinkhorn" loss ('blur':σ=0.01, 'scaling':0.9)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01, scaling=0.9)
    return loss(x, y)
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

class KMeansClustering(BaseTransform):
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def fit(self, data):
        pos = data.pos
        x = data.x

        N = data.num_nodes
        k = N // self.num_clusters

        pos_flattened = pos.view(-1, pos.size(-1)).numpy()

        kmeans = KMeans(n_clusters=k, n_init=3)
        self.labels = kmeans.fit_predict(pos_flattened)
        self.labels = torch.from_numpy(self.labels)  # Convert labels to torch.Tensor
        self.centroids_pos = torch.zeros(k, pos.size(-1))
        self.centroids_x = torch.zeros(k, x.size(-1))

        for node_idx, cluster_idx in enumerate(self.labels):
            self.centroids_pos[cluster_idx] += pos[node_idx]
            self.centroids_x[cluster_idx] += x[node_idx]

        for cluster_idx in range(k):
            indices = torch.nonzero(self.labels == cluster_idx).view(-1)
            count = indices.size(0)

            self.centroids_pos[cluster_idx] /= count
            self.centroids_x[cluster_idx] /= count

    def __call__(self, data):
        pos = data.pos
        x = data.x

        # Assign the precomputed centroids and labels
        data.x = self.centroids_x
        data.x_full = x
        data.pos = self.centroids_pos
        data.pos_full = pos
        data.cluster_labels = self.labels

        return data

class MNISTSuperpixels(InMemoryDataset):
    url = 'https://data.pyg.org/datasets/MNISTSuperpixels.zip'

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        cluster_k: int = None,
        **kwargs,
    ):
        self.cluster_k = cluster_k  # Store cluster_k for later use
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> str:
        return 'MNISTSuperpixels.pt'

    @property
    def processed_file_names(self) -> List[str]:
        if self.cluster_k is None:
            return ['train_data.pt', 'test_data.pt']
        else:
            return [f'train_data_k{self.cluster_k}.pt', f'test_data_k{self.cluster_k}.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        inputs = torch.load(self.raw_paths[0])
        for i in range(len(inputs)):
            data_list = [Data(**data_dict) for data_dict in inputs[i]]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            if self.cluster_k is not None:
                with tqdm(total=len(data_list), desc=f'Cluster K={self.cluster_k}') as pbar:
                    for j in range(len(data_list)):
                        cluster_transform = KMeansClustering(num_clusters=self.cluster_k)
                        cluster_transform.fit(data_list[j])
                        data_list[j] = cluster_transform(data_list[j])
                        pbar.update(1)

            torch.save(self.collate(data_list), self.processed_paths[i])

def get_datasets(data_dir, batch_size, radius, subgraph_dict=None):
    cluster_k = 3
    transforms = []
    transforms.append(RadiusGraph(radius))
    if subgraph_dict is not None:
        subgraph_mode = subgraph_dict.get("mode", None)
        transforms.append(Graph_to_Subgraph(mode=subgraph_mode))
    transforms = Compose(transforms)
    # TODO: RESCALE THE DATASET BACK TO THE ORIGINAL SIZE
    train_val_set = MNISTSuperpixels(root=data_dir, transform=transforms, train=True, cluster_k=cluster_k)
    # split train into train and val sets by taking the last 10% of the training set
    train_set = train_val_set[:int(len(train_val_set) * 0.9)]
    train_set = train_set[:1000]
    val_set = train_val_set[int(len(train_val_set) * 0.9):]
    val_set = val_set[:1000]
    test_set = MNISTSuperpixels(root=data_dir, transform=transforms, train=False, cluster_k=cluster_k)
    # print which transforms are we using
    print("Transforms: ", transforms)
    #assert len(train_set) + len(val_set) == len(train_val_set)

    train_loader = tg.loader.DataLoader(train_set, batch_size=batch_size, shuffle=True, )
    val_loader = tg.loader.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = tg.loader.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class MNISTSuperpixelsUpscale(pl.LightningModule):
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

        self.train_sinkhorn_loss = []
        self.val_sinkhorn_loss = []

    def forward(self, graph):
        return self.model(graph)

    def training_step(self, graph):
        graph = graph.to(self.device)
        # add gaussian noise to the positions of the graph
        graph.pos = graph.pos + torch.randn(graph.pos.shape).to(self.device) * 0.01
        superpixel_pos, superpixel_h = self(graph)
        # Check if the gradients of the model are nan. Go through all parameters of the model and print which one has nan gradient

        batch_size = graph.batch.max() + 1
        # Pack the positions and the superpixel_h into a tensor [B, N, 2]
        superpixel_pos = torch.stack([superpixel_pos[i] for i in graph.batch])
        true_pos = torch.stack([graph.pos_full[i] for i in graph.batch])
        # assert that true pos and superpixel pos have the same shape
        assert superpixel_pos.shape == true_pos.shape
        # print the shapes

        loss = sinkhorn_loss(superpixel_pos, true_pos)

        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.train_sinkhorn_loss.append(loss.item())
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        return loss

    def on_train_epoch_end(self):
        avg_train_loss = torch.mean(torch.tensor(self.train_sinkhorn_loss))
        self.log("train_loss", avg_train_loss, prog_bar=True)
        self.train_sinkhorn_loss = []

    def validation_step(self, graph, batch_idx):
        graph = graph.to(self.device)
        superpixel_pos, superpixel_h = self(graph)
        superpixel_pos = torch.stack([superpixel_pos[i] for i in graph.batch])
        true_pos = torch.stack([graph.pos_full[i] for i in graph.batch])
        loss = sinkhorn_loss(superpixel_pos, true_pos)
        self.val_sinkhorn_loss.append(loss.item())

        return loss

    def on_validation_epoch_end(self):
        avg_val_loss = torch.mean(torch.tensor(self.val_sinkhorn_loss))
        self.log("val_loss", avg_val_loss, prog_bar=True)
        self.val_sinkhorn_loss = []

    def test_step(self, graph, batch_idx):
        graph = graph.to(self.device)
        superpixel_pos, superpixel_h = self(graph)
        loss = sinkhorn_loss(superpixel_pos, graph.pos_full)
        self.test_metric = loss.item()

        return loss

    def on_test_epoch_end(self):
        avg_test_loss = torch.mean(torch.tensor(self.test_metric))
        self.log("test_loss", avg_test_loss, prog_bar=True)
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
