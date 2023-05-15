#!/bin/bash

# Check if dataset name is provided as an argument
if [ -z "$1" ]; then
    echo "Please provide the dataset name as an argument."
    exit 1
fi

# Set the dataset name
dataset_name="$1"

# Set the source and destination paths
source_path="thadzi@snellius.surf.nl:/home/thadzi/GitHub/FractalMessagePassing/logs/$dataset_name"
dest_path="downloaded_logs/$dataset_name"

# Copy the logs folder from the remote server to the local machine
echo "Copying logs folder from remote server..."
scp -r "$source_path" "$dest_path"

# Start TensorBoard on the local machine
echo "Starting TensorBoard on the local machine..."
tensorboard --logdir="$dest_path"
