#!/bin/bash

# Run the Python scripts one by one
python lightning_train.py "configs/mnist/EGNN.yaml"
python lightning_train.py "configs/mnist/Transformer_EGNN_v2.yaml"
python lightning_train.py "configs/mnist/MPNN.yaml"
python lightning_train.py "configs/mnist/Transformer_MPNN.yaml"
# Add more Python scripts here if needed
