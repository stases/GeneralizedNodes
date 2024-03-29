import sys
from datetime import datetime

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.gnn.networks import *
from misc.train_qm9_debug import train_qm9_model
from trainers.train_md17 import train_md17_model
from trainers.train_md17_lightning import MD17Model
from trainers.train_qm9_lightning import QM9Model
from trainers.train_MNIST_lightning import MNISTModel
from trainers.train_MNIST_upscale_lightning import MNISTSuperpixelsUpscale
from trainers.train_wikics import train_wikics_model


import warnings

# Mute the specific warning by category and message
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.*")

#####################
#  Helper functions #
MODEL_MAP = {
    "fractalnet": FractalNet,
    "net": Net,
    "transformernet": TransformerNet,
    "MPNN": MPNN,
    "Simple_MPNN": Simple_MPNN,
    "Simple_Transformer_MPNN": Simple_Transformer_MPNN,
    "Transformer_MPNN": Transformer_MPNN,
    "EGNN": EGNN,
    "EGNN_Full": EGNN_Full,
    "Fractal_EGNN": Fractal_EGNN,
    "Fractal_EGNN_v2": Fractal_EGNN_v2,
    "Transformer_EGNN": Transformer_EGNN,
    "Transformer_EGNN_v2": Transformer_EGNN_v2,
    "Superpixel_EGNN": Superpixel_EGNN,
    # Add more models here
}

TRAINER_MAP = {
    "qm9": QM9Model,
    "md17": MD17Model,
    "mnist": MNISTModel,
    "mnist_upscale": MNISTSuperpixelsUpscale,
    "wikics": train_wikics_model,
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
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    # Add more schedulers here
}

DEVICE_MAP = {
    "cuda": "cuda",
    "cpu": "cpu"
    # Add more devices here
}
#####################

#####################
#  Config loading    #
config_file = sys.argv[1]
with open(config_file, "r") as f:
    config = yaml.safe_load(f)
#####################

#####################
#  Config parsing    #
trainer_name = config.get("trainer", "qm9")
model_arch = config.get("model_arch", "gnn")
subgraph_dict = config.get("subgraph", {})
learning_rate = config.get("learning_rate", 0.001)
batch_size = config.get("batch_size", 32)
epochs = config.get("epochs", 10)
data_dir = config.get("data_dir", "default_data_dir")
model_dir = config.get("model_dir", "default_model_dir")
log_dir = config.get("log_dir", "default_log_dir")
device = config.get("device", "cuda")
#####################

#####################
#   Creating dirs   #
# check if the directory exists, if not create it
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
#####################

#####################
# Trainer loading   #
trainer_class = TRAINER_MAP.get(trainer_name, None)
if trainer_class is None:
    raise ValueError(f"Invalid trainer value: {trainer_name}")
#####################

#####################
# Generating run ID #
#config["run_id"] = np.random.randint(0, 1000000)
config["run_id"] = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
config["model_name"] = f"{model_arch}_{config['run_id']}"
'''# save the current config to the model directory
with open(f"{model_dir}/{config['model_name']}.yaml", "w") as f:
    yaml.dump(config, f)'''
# save it in the log directory as well
with open(f"{log_dir}/{config['model_name']}.yaml", "w") as f:
    yaml.dump(config, f)
#####################

#####################
#  Model loading     #
# Load the model class or function based on the model_arch key
model_class = MODEL_MAP.get(model_arch, None)
if model_class is None:
    raise ValueError(f"Invalid model_arch value: {model_arch}")

# Instantiate the model using kwargs from the YAML configuration file
model = model_class(**config)
model = model.to(device)
#####################


#####################
#  Training loop    #
print(f"Training {model_arch} on {trainer_name} dataset. Run ID: {config['run_id']}.")
trainer = trainer_class(model=model, **config)
#####################