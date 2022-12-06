################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-17
################################################################################
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn

class GNN(nn.Module):
    """implements a graphical neural network in pytorch. In particular, we will use pytorch geometric's nn_conv module so we can apply a neural network to the edges.
    """

    def __init__(
            self,
            n_node_features: int,
            n_edge_features: int,
            n_hidden: int,
            n_output: int,
            num_convolution_blocks: int,
    ) -> None:
        """create the gnn

        Args:
            n_node_features: input features on each node
            n_edge_features: input features on each edge
            n_hidden: hidden features within the neural networks (embeddings, nodes after graph convolutions, etc.)
            n_output: how many output features
            num_convolution_blocks: how many blocks convolutions should be performed. A block may include multiple convolutions

        TODO:
        - define a GNN which has the following structure: node embedding -> [ReLU -> RGCNConv -> ReLU -> MFConv] x num_convs -> Add-Pool -> Linear -> ReLU -> Linear
        - One the data has been pooled, it may be beneficial to apply another MLP on the pooled data before predicing the output.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()

        layers = []
        in_channels = n_node_features
        not_only = False
        self.embed = nn.Linear(n_node_features, n_hidden[0])
        for l_idx, hidden_dim in enumerate(n_hidden):
            layers += [
                nn.Identity() if l_idx == 0 else nn.ReLU(),
                geom_nn.RGCNConv(in_channels=n_hidden[l_idx],
                                 out_channels=n_hidden[l_idx + 1],
                                 num_relations=n_edge_features),
                nn.ReLU(),
                geom_nn.MFConv(in_channels=n_hidden[l_idx],
                               out_channels=n_hidden[l_idx + 1],
                               max_degree=10) if not_only else nn.Identity()
            ]
            if (l_idx == len(n_hidden) - 2):
                break
        out_channels = n_hidden[-1]
        self.GNN = nn.ModuleList(layers)
        self.linear_1 = nn.Linear(out_channels, out_channels)
        self.linear_2 = nn.Linear(out_channels, 1)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            edge_attr: edge attributes (pytorch geometric notation)
            batch_idx: Index of batch element for each node

        Returns:
            prediction

        TODO: implement the forward pass being careful to apply MLPs only where they are allowed!

        Hint: remember to use global pooling.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.embed(x)
        edge_index = edge_index.type(torch.int64)
        for layer in self.GNN:
            if (isinstance(layer, geom_nn.RGCNConv)):
                x = layer(x, edge_index, edge_attr)

            elif (isinstance(layer, geom_nn.MessagePassing)):
                x = layer(x, edge_index)

            else:
                x = layer(x)
        x = geom_nn.global_add_pool(x, batch_idx)
        x = F.relu(self.linear_1(x))
        out = self.linear_2(x)
        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device


class GNN_no_rel(nn.Module):
    """implements a graphical neural network in pytorch. In particular, we will use pytorch geometric's nn_conv module so we can apply a neural network to the edges.
    """

    def __init__(
            self,
            n_node_features: int,
            n_edge_features: int,
            n_hidden: int,
            n_output: int,
            num_convolution_blocks: int,
    ) -> None:
        """create the gnn

        Args:
            n_node_features: input features on each node
            n_edge_features: input features on each edge
            n_hidden: hidden features within the neural networks (embeddings, nodes after graph convolutions, etc.)
            n_output: how many output features
            num_convolution_blocks: how many blocks convolutions should be performed. A block may include multiple convolutions

        TODO:
        - define a GNN which has the following structure: node embedding -> [ReLU -> RGCNConv -> ReLU -> MFConv] x num_convs -> Add-Pool -> Linear -> ReLU -> Linear
        - One the data has been pooled, it may be beneficial to apply another MLP on the pooled data before predicing the output.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()

        layers = []
        in_channels, out_channels = n_node_features, n_hidden

        self.embed = nn.Linear(n_node_features, n_hidden)
        for l_idx in range(num_convolution_blocks):
            layers += [
                nn.Identity() if l_idx == 0 else nn.ReLU(),
                geom_nn.MFConv(in_channels=n_hidden,
                               out_channels=n_hidden,
                               max_degree=10)
            ]
        self.GNN = nn.ModuleList(layers)
        self.linear_1 = nn.Linear(out_channels, out_channels)
        self.linear_2 = nn.Linear(out_channels, 1)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            edge_attr: edge attributes (pytorch geometric notation)
            batch_idx: Index of batch element for each node

        Returns:
            prediction

        TODO: implement the forward pass being careful to apply MLPs only where they are allowed!

        Hint: remember to use global pooling.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.embed(x)
        edge_index = edge_index.type(torch.int64)
        for layer in self.GNN:
            if (isinstance(layer, geom_nn.RGCNConv)):
                x = layer(x, edge_index, edge_attr)

            elif (isinstance(layer, geom_nn.MessagePassing)):
                x = layer(x, edge_index)

            else:
                x = layer(x)
        x = geom_nn.global_add_pool(x, batch_idx)
        x = F.relu(self.linear_1(x))
        out = self.linear_2(x)
        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
