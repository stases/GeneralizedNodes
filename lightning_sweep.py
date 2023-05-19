import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from models.gnn.networks import *
from trainers.train_qm9_debug import train_qm9_model
from trainers.train_md17 import train_md17_model
from trainers.train_md17_lightning import MD17Model


#####################
#  Training loop    #

egnn_sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'Energy valid MAE',
      'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0001,
            'max': 0.0008,
        },
        'depth': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 7,
        },
        'epochs': {
            'value': 10
        },
        'norm': {
            'values': ['layer', 'batch', 'none']
        },
        'batch_size': {
            'values': [4,8,16,32]
        },
        'RFF_dim': {
            'values': [32,48,64,96,128]
        },
        'RFF_sigma': {
            'min': 1.0,
            'max': 15.0
        },
        'hidden_features': {
            'values': [32,48,64,96,128]
        },
        'node_features': {
            'value': 9
        },
        'out_features': {
            'value': 1
        },
        'name': {
            'value': "aspirin CCSD"
        },
        'data_dir': {
            'value': "./data/md17"
        },
    }
}

fractal_egnn_sweep_config = {}

SWEEP_TARGET = "egnn" # or "fractal_egnn"
if SWEEP_TARGET == "egnn":
    sweep_config = egnn_sweep_config
    project_name = "egnn_sweep"
elif SWEEP_TARGET == "fractal_egnn":
    sweep_config = fractal_egnn_sweep_config
    project_name = "fractal_egnn_sweep"

sweep_id = wandb.sweep(sweep_config, project=project_name)

def train():
    wandb_logger = WandbLogger()
    with wandb.init() as run:
        config = {**run.config, 'node_features': 9,
                  'out_features': 1,
                  'data_dir': "./data/md17",
                  'name': "aspirin CCSD"}
        if SWEEP_TARGET == "egnn":
            model = EGNN(**config)
        elif SWEEP_TARGET == "fractal_egnn":
            model = Fractal_EGNN(**config)

        trainer = pl.Trainer(max_epochs=config['epochs'], logger=wandb_logger, accelerator='gpu', devices=1)
        lightning_model = MD17Model(model, **config)
        trainer.fit(lightning_model)

wandb.agent(sweep_id, function=train)