import sys
import yaml
from models.gnn.networks import FractalNet, FractalNetShared, GNN, GNN_no_rel, Net, TransformerNet
from train import train_qm9

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
    "qm9": train_qm9,
    # Add more trainers here
}

# Load the YAML configuration file
config_file = sys.argv[1]
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# Extract the necessary information
dataset = config.get("dataset", "qm9")
model_arch = config.get("model_arch", "gnn")
learning_rate = config.get("learning_rate", 0.001)
batch_size = config.get("batch_size", 32)
train_data = config.get("train_data", "default_train_data")
val_data = config.get("val_data", "default_val_data")
test_data = config.get("test_data", "default_test_data")
model_dir = config.get("model_dir", "default_model_dir")
log_dir = config.get("log_dir", "default_log_dir")

# Load the model class or function based on the model_arch key
model_class = MODEL_MAP.get(model_arch, None)
if model_class is None:
    raise ValueError(f"Invalid model_arch value: {model_arch}")

# Add model to the config dictionary
config["model"] = model_class

# Instantiate the model using kwargs from the YAML configuration file
model = model_class(**config)

# Load the trainer class or function based on the dataset key
trainer_class = TRAINER_MAP.get(dataset, None)
if trainer_class is None:
    raise ValueError(f"Invalid dataset value: {dataset}")

# Train the model using kwargs from the YAML configuration file
trainer_class(model, **config)

# Print the extracted information as a space-separated list
print(f"{model_arch} {learning_rate} {batch_size} {train_data} {val_data} {test_data} {model_dir} {log_dir}")
