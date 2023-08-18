import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from models.gnn.networks import *
from trainers.train_md17_lightning import MD17Model


#####################
#  Training loop    #

egnn_sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'Force valid MAE',
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
            'value': 150
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

fractal_egnn_sweep_config = {
    'method': 'random',
    'metric': {
      'name': 'Force valid MAE',
      'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0003,
            'max': 0.0008,
        },
        'depth': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 3,
        },
        'model_type': {
            'values': ['fractal_egnn', 'fractal_egnn_v2']
        },
        'epochs': {
            'value': 1
        },
        'norm': {
            'values': ['layer', 'batch', 'none']
        },
        'mask': {
            'values': [True, False]
        },

        'batch_size': {
            'values': [4,8]
        },
        'RFF_dim': {
            'values': [32,48,64,96]
        },
        'RFF_sigma': {
            'values': [0.1, 1, 4, 8, 15]
        },
        'hidden_features': {
            'values': [32,48,64,96]
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

SWEEP_TARGET = "fractal_egnn" # or "fractal_egnn"

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
            config['subgraph_dict'] = {'mode': "fractal"}
            if config['model_type'] == 'fractal_egnn':
                model = Fractal_EGNN(**config)
            elif config['model_type'] == 'fractal_egnn_v2':
                model = Fractal_EGNN_v2(**config)

        trainer = pl.Trainer(max_epochs=config['epochs'], logger=wandb_logger, accelerator='gpu', devices=1)
        lightning_model = MD17Model(model, **config)
        trainer.fit(lightning_model)

wandb.agent(sweep_id, function=train)