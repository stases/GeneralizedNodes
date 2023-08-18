#!/bin/bash

# Define the path to the config
CONFIG_PATH="configs/qm9/Transformer_EGNN_v2_light.yaml"

# Loop through the desired range
for i in {0..10}
do
    echo "Running with LABEL_INDEX=$i"
    python lightning_train.py $CONFIG_PATH --LABEL_INDEX $i
done

echo "All tasks completed!"
