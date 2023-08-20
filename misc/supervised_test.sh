#!/bin/bash

# Define the path to the config
CONFIG_PATH_1="configs/qm9/EGNN_light.yaml"
CONFIG_PATH_2="configs/supervised_qm9/RelEGNN.yaml"

python lightning_train.py $CONFIG_PATH_1
python lightning_train.py $CONFIG_PATH_2
echo "All tasks completed!"
