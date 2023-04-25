#!/bin/bash

# Change to the directory containing this script
cd "$(dirname "$0")"

# Loop through all YAML config files in the configs directory
for config_file in configs/*.yaml; do
    # Run train.py for each config file
    python train.py "$config_file"
done