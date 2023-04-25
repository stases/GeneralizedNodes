import sys
import yaml
import numpy as np
from models.gnn.networks import FractalNet, FractalNetShared, GNN, GNN_no_rel, Net, TransformerNet
from trainers.train_qm9 import train_qm9_model
import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler


# Define a mapping between model names and the corresponding classes or functions
MODEL_MAP = {
    "fractalnet": FractalNet,
    "net": Net,
    "transformernet": TransformerNet,
    "gnn": GNN,
    "gnn_no_rel": GNN_no_rel,
    "fractalnet_shared": FractalNetShared,
    # Add more models here
}

TRAINER_MAP = {
    "qm9": train_qm9_model,
    # Add more trainers here
}


OPTIMIZER_MAP = {
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    # Add more optimizers here
}

CRITERION_MAP = {
    "MSELoss": nn.MSELoss,
    "L1Loss": nn.L1Loss,
    # Add more criteria here
}

SCHEDULER_MAP = {
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    "StepLR": lr_scheduler.StepLR,
    # Add more schedulers here
}

DEVICE_MAP = {
    "cuda": "cuda",
    "cpu": "cpu"
    # Add more devices here
}

# Load the YAML configuration file
config_file = sys.argv[1]
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# Extract the necessary information
trainer_name = config.get("trainer", "qm9")
model_arch = config.get("model_arch", "gnn")
subgraph = config.get("subgraph", False)
learning_rate = config.get("learning_rate", 0.001)
batch_size = config.get("batch_size", 32)
data_dir = config.get("data_dir", "default_data_dir")
model_dir = config.get("model_dir", "default_model_dir")
log_dir = config.get("log_dir", "default_log_dir")
device = config.get("device", "cuda")

# Initiate the model
# Load the model class or function based on the model_arch key
model_class = MODEL_MAP.get(model_arch, None)
if model_class is None:
    raise ValueError(f"Invalid model_arch value: {model_arch}")

# Instantiate the model using kwargs from the YAML configuration file
model = model_class(**config)
model = model.to(device)

# Load the optimizer class or function based on the optimizer key
optimizer_dict = config.get("optimizer", {})
optimizer_name = optimizer_dict.get("name", "Adam")
optimizer_kwargs = optimizer_dict.get("kwargs", {})
if "eps" in optimizer_kwargs:
    optimizer_kwargs["eps"] = float(optimizer_kwargs["eps"])

optimizer_class = OPTIMIZER_MAP.get(optimizer_name, None)
if optimizer_class is None:
    raise ValueError(f"Invalid optimizer value: {optimizer_name}")

# Instantiate the optimizer using kwargs from the YAML configuration file
optimizer = optimizer_class(model.parameters(), lr=learning_rate, **optimizer_kwargs)
config["optimizer"] = optimizer


# Load the criterion class or function based on the criterion key
criterion_dict = config.get("criterion", {})
criterion_name = criterion_dict.get("name", "MSELoss")
criterion_kwargs = criterion_dict.get("kwargs", {})

criterion_class = CRITERION_MAP.get(criterion_name, None)
if criterion_class is None:
    raise ValueError(f"Invalid criterion value: {criterion_name}")

# Instantiate the criterion using kwargs from the YAML configuration file
criterion = criterion_class(**criterion_kwargs)
config["criterion"] = criterion

# Load the scheduler class or function based on the scheduler key
scheduler_dict = config.get("scheduler", {})
scheduler_name = scheduler_dict.get("name", "ReduceLROnPlateau")
scheduler_kwargs = scheduler_dict.get("kwargs", {})

scheduler_class = SCHEDULER_MAP.get(scheduler_name, None)
if scheduler_class is None:
    raise ValueError(f"Invalid scheduler value: {scheduler_name}")

# Instantiate the scheduler using kwargs from the YAML configuration file
scheduler = scheduler_class(optimizer, **scheduler_kwargs)
config["scheduler"] = scheduler

# Load the device based on the device key
device_name = DEVICE_MAP.get(device, None)
if device_name is None:
    raise ValueError(f"Invalid device value: {device}")
# Set the device
device = torch.device(device_name)

# Generate a random number that will uniquely identify this run
config["run_id"] = np.random.randint(0, 1000000)
config["model_name"] = f"{model_arch}_{config['run_id']}"
# save the current config to the model directory
with open(f"{model_dir}/{config['model_name']}.yaml", "w") as f:
    yaml.dump(config, f)

# Train the model using kwargs from the YAML configuration file
# Load the trainer class or function based on the dataset key
trainer_class = TRAINER_MAP.get(trainer_name, None)
if trainer_class is None:
    raise ValueError(f"Invalid trainer value: {trainer_name}")

# Instantiate the trainer using kwargs from the YAML configuration file
trainer = trainer_class(model=model, **config)

# Print the extracted information as a space-separated list
print(f"{model_arch} {learning_rate} {batch_size} {train_data} {val_data} {test_data} {model_dir} {log_dir}")