import os
from typing import Callable, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric as tg
from geomloss import SamplesLoss
from sklearn.cluster import KMeans
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms import RadiusGraph, Compose
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import wandb
from utils.transforms import Graph_to_Subgraph
import wandb
from PIL import Image
import io
import torch_geometric.data
import torch
from torch_geometric.utils import to_networkx

def visualize_sample(input_graph, output_graph, target_graph, mode):
    fig, axs = plt.subplots(1, 3, figsize=(36, 12))

    # For the input graph
    sample_graph = input_graph
    sample_graph.pos -= sample_graph.pos.min()
    sample_graph.pos = sample_graph.pos/sample_graph.pos.max() * 2 - 1
    pos_dict = {}
    for i, p in enumerate(sample_graph.pos):
        pos_dict[i] = p.detach().numpy() * np.array([1, -1])
    g = to_networkx(sample_graph, to_undirected=True)
    nx.draw_networkx_nodes(g,
                           node_size=500,
                           node_color=sample_graph.x.detach().cpu().numpy(),
                           node_shape=r'$\circ$',
                           pos=pos_dict,
                           cmap='Purples',
                           ax=axs[0])
    nx.draw_networkx_edges(g, edge_color='r', alpha=0.5, pos=pos_dict, ax=axs[0])
    axs[0].set_title('Input Graph', fontsize=25)

    # For the output graph
    sample_graph = output_graph
    sample_graph.pos -= sample_graph.pos.min()
    sample_graph.pos = sample_graph.pos/sample_graph.pos.max() * 2 - 1
    pos_dict = {}
    for i, p in enumerate(sample_graph.pos):
        pos_dict[i] = p.detach().numpy() * np.array([1, -1])
    g = to_networkx(sample_graph, to_undirected=True)
    nx.draw_networkx_nodes(g,
                           node_size=500,
                           node_color=sample_graph.x.detach().cpu().numpy(),
                           node_shape=r'$\circ$',
                           pos=pos_dict,
                           cmap='Purples',
                           ax=axs[1])
    nx.draw_networkx_edges(g, edge_color='r', alpha=0.5, pos=pos_dict, ax=axs[1])
    axs[1].set_title('Output Graph', fontsize=25)

    # For the target graph
    sample_graph = target_graph
    sample_graph.pos -= sample_graph.pos.min()
    sample_graph.pos = sample_graph.pos/sample_graph.pos.max() * 2 - 1
    pos_dict = {}
    for i, p in enumerate(sample_graph.pos):
        pos_dict[i] = p.detach().numpy() * np.array([1, -1])
    g = to_networkx(sample_graph, to_undirected=True)
    nx.draw_networkx_nodes(g,
                           node_size=500,
                           node_color=sample_graph.x.detach().cpu().numpy(),
                           node_shape=r'$\circ$',
                           pos=pos_dict,
                           cmap='Purples',
                           ax=axs[2])
    nx.draw_networkx_edges(g, edge_color='r', alpha=0.5, pos=pos_dict, ax=axs[2])
    axs[2].set_title('Target Graph', fontsize=25)
    #plt.show()
    # Convert the figure to an image
    wandb_image = wandb.Image(fig)

    # Log the image using WandB
    if mode == "train":
        wandb.log({"Train Sample Image": wandb_image})
    elif mode == "val":
        wandb.log({"Validation Sample Image": wandb_image})
    elif mode == "train_specific":
        wandb.log({"Train Specific Sample Image": wandb_image})
    elif mode == "val_specific":
        wandb.log({"Validation Specific Sample Image": wandb_image})
    plt.close(fig)

def extract_graph(data, graph_id):
    device = 'cpu'

    # Get mask of nodes that belong to the graph_id
    node_mask = data.batch == graph_id
    node_mask = node_mask.to(device)
    # Extract node attributes
    x = None
    pos = None
    if data.x is not None:
        x = data.x[node_mask]
    if data.pos is not None:
        pos = data.pos[node_mask]

    x = x.to(device)
    pos = pos.to(device)
    
    # Create a mapping from old node indices to new node indices
    node_index_mapping = torch.full((data.batch.size(0), ), -1, dtype=torch.long, device=device).to(device)
    node_index_mapping[node_mask] = torch.arange(node_mask.sum().item(), dtype=torch.long)

    # Get mask of edges that belong to the graph_id
    node_mask = node_mask.to(device)
    data.edge_index = data.edge_index.to(device)
    edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]].to(device)

    # Extract edge attributes
    edge_index = None
    if data.edge_index is not None:
        edge_index = node_index_mapping[data.edge_index[:, edge_mask]].to(device)

    # Create single graph data
    single_graph_data = torch_geometric.data.Data(x=x, edge_index=edge_index, pos=pos)

    return single_graph_data

def prepare_sample(batch, superpixel_pos, superpixel_h, radius=8):
    input_data = Data(x=batch.x[batch.ground_node][:, 0].unsqueeze(-1), pos=batch.pos[batch.ground_node],
                      batch=batch.batch[batch.ground_node], y=batch.y)
    input_data = RadiusGraph(radius)(input_data)

    output_data = Data(x=superpixel_h, pos=superpixel_pos, batch=batch.batch[~batch.ground_node], y=batch.y)
    output_data = RadiusGraph(radius)(output_data)

    target_data = Data(pos=batch.pos_full, x=batch.x_full, edge_index=batch.edge_index,
                       batch=batch.batch[~batch.ground_node], y=batch.y)
    target_data = RadiusGraph(radius)(target_data)
    return input_data, output_data, target_data

def sinkhorn_loss(x, y):
    # "sinkhorn" loss ('blur':σ=0.01, 'scaling':0.9)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01, scaling=0.9)
    return loss(x, y)

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

class MNISTSuperpixelsUpscale(pl.LightningModule):
    def __init__(self, model, data_dir, radius, batch_size, warmup_epochs, subgraph_dict=None, log_img=True, **kwargs):
        super().__init__()
        self.model = model
        print(self.model)
        self.data_dir = data_dir
        self.radius = radius
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.subgraph_dict = subgraph_dict
        self.learning_rate = kwargs["learning_rate"]
        self.graphs_to_plot = 1
        self.train_sinkhorn_loss = []
        self.val_sinkhorn_loss = []
        self.log_img = log_img
        self.current_train_data = None
        self.specific_train_data = None
        self.current_val_data = None
        self.specific_val_data = None
        self.specific_train_data_assigned = False
        self.specific_val_data_assigned = False

    def forward(self, graph):
        return self.model(graph)

    def training_step(self, batch):
        batch = batch.to(self.device)
        graph = batch
        # add gaussian noise to the positions of the graph
        #graph.pos = graph.pos + torch.randn(graph.pos.shape).to(self.device) * 0.01
        superpixel_pos, superpixel_h = self(graph)

        # LOSS
        diff = graph.pos_full - superpixel_pos
        squared_diff = diff ** 2
        loss_pos = torch.mean(squared_diff)
        loss_h = torch.mean(torch.square(graph.x_full - superpixel_h))
        #loss = loss_pos + loss_h
        loss = loss_pos

        self.current_train_data = {'input': graph,
                             'output_pos': superpixel_pos,
                             'output_h': superpixel_h,
                             }
        if self.specific_train_data_assigned is False:
            self.specific_train_data = graph
            self.specific_train_data_assigned = True

        # Pack the positions and the superpixel_h into a tensor [B, N, 2]
        superpixel_pos = torch.stack([superpixel_pos[i] for i in graph.batch])
        true_pos = torch.stack([graph.pos_full[i] for i in graph.batch])
        # assert that true pos and superpixel pos have the same shape
        assert superpixel_pos.shape == true_pos.shape
        # LOSS TODO SINKHORN
        #loss = sinkhorn_loss(superpixel_pos, true_pos)

        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.train_sinkhorn_loss.append(loss.item())
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        return loss

    def on_train_epoch_end(self):
        avg_train_loss = torch.mean(torch.tensor(self.train_sinkhorn_loss))
        self.log("train_loss", avg_train_loss, prog_bar=True)
        self.train_sinkhorn_loss = []
        if self.current_train_data is not None:
            for idx in range(self.graphs_to_plot):
                batch = self.current_train_data['input']
                superpixel_pos = self.current_train_data['output_pos']
                superpixel_h = self.current_train_data['output_h']

                input_data, output_data, target_data = prepare_sample(batch, superpixel_pos, superpixel_h)

                input_graph = extract_graph(input_data, idx)
                output_graph = extract_graph(output_data, idx)
                target_graph = extract_graph(target_data, idx)
                if self.log_img:
                    visualize_sample(input_graph, output_graph, target_graph, "train")
        if self.log_img:
            batch = self.specific_train_data
            superpixel_pos, superpixel_h = self(batch)
            input_data, output_data, target_data = prepare_sample(batch, superpixel_pos, superpixel_h)

            input_graph = extract_graph(input_data, 0)
            output_graph = extract_graph(output_data, 0)
            target_graph = extract_graph(target_data, 0)
            visualize_sample(input_graph, output_graph, target_graph, "train_specific")

    def validation_step(self, graph, batch_idx):
        graph = graph.to(self.device)
        #graph.pos + torch.randn(graph.pos.shape).to(self.device) * 0.01
        superpixel_pos, superpixel_h = self(graph)
        self.current_val_data = {'input': graph,
                                'output_pos': superpixel_pos,
                                'output_h': superpixel_h,
                                }
        if self.specific_val_data_assigned is False:
            self.specific_val_data = graph
            self.specific_val_data_assigned = True

        superpixel_pos = torch.stack([superpixel_pos[i] for i in graph.batch])
        true_pos = torch.stack([graph.pos_full[i] for i in graph.batch])
        loss = sinkhorn_loss(superpixel_pos, true_pos)
        self.val_sinkhorn_loss.append(loss.item())

        return loss

    def on_validation_epoch_end(self):
        avg_val_loss = torch.mean(torch.tensor(self.val_sinkhorn_loss))
        self.log("val_loss", avg_val_loss, prog_bar=True)
        self.val_sinkhorn_loss = []
        if self.current_val_data is not None:
            for idx in range(self.graphs_to_plot):
                batch = self.current_val_data['input']
                superpixel_pos = self.current_val_data['output_pos']
                superpixel_h = self.current_val_data['output_h']
                input_data, output_data, target_data = prepare_sample(batch, superpixel_pos, superpixel_h)

                input_graph = extract_graph(input_data, idx)
                output_graph = extract_graph(output_data, idx)
                target_graph = extract_graph(target_data, idx)
                if self.log_img:
                    visualize_sample(input_graph, output_graph, target_graph, "val")
        if self.log_img:
            batch = self.specific_val_data
            superpixel_pos, superpixel_h = self(batch)

            input_data, output_data, target_data = prepare_sample(batch, superpixel_pos, superpixel_h)

            input_graph = extract_graph(input_data, 0)
            output_graph = extract_graph(output_data, 0)
            target_graph = extract_graph(target_data, 0)
            visualize_sample(input_graph, output_graph, target_graph, "val_specific")


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
