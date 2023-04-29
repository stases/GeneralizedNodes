#!/bin/bash

# Set the source and destination paths
source_path="thadzi@snellius.surf.nl:/home/thadzi/GitHub/FractalMessagePassing/logs"
dest_path="downloaded_logs/"

# Copy the logs folder from the remote server to the local machine
echo "Copying logs folder from remote server..."
scp -r "$source_path" "$dest_path"

# Start TensorBoard on the local machine
echo "Starting TensorBoard on local machine..."
tensorboard --logdir="$dest_path"
