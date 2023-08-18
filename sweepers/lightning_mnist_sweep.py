import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from models.gnn.networks import *
from trainers.train_MNIST_lightning import MNISTModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

#####################
#  Training loop    #

transformer_sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'valid acc',
      'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0003,
            'max': 0.001,
        },
        'depth': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 5,
        },
        'epochs': {
            'value': 80
        },
        'radius': {
            'values': [8]
        },
        'warmup_epochs': {
            'values': [3]
        },
        'edge_features': {
            'values': [0]
        },
        'mask': {
            'values': [True, False]
        },
        'only_ground': {
            'values': [True, False]
        },
        'only_sub': {
            'values': [True, False]
        },
        'ascend_depth': {
            'values': [0, 2]
        },
        'num_heads': {
            'values': [1, 2, 4]
        },
        'num_ascend_heads': {
            'values': [1, 2, 4]
        },
        'residual': {
            'values': [True, False]
        },
        'transformer_size': {
            'values': [2,4,8]
        },
        'norm': {
            'values': ['layer', 'none']
        },
        'batch_size': {
            'values': [64,128]
        },
        'hidden_features': {
            'values': [32,48,64]
        },
        'node_features': {
            'value': 1
        },
        'out_features': {
            'value': 10
        },
        'data_dir': {
            'value': "./data/MNIST"
        },
    }
}

SWEEP_TARGET = "transformer"

if SWEEP_TARGET == "transformer":
    sweep_config = transformer_sweep_config
    project_name = "MNIST_sweep_transformer"

sweep_id = wandb.sweep(sweep_config, project=project_name)

def train():
    wandb_logger = WandbLogger()
    with wandb.init() as run:
        transformer_size = run.config['transformer_size']
        mode = "transformer_" + str(transformer_size)
        subgraph_dict = {'mode': mode}
        node_features = 1 + transformer_size
        config = {**run.config, 'node_features': node_features, 'subgraph_dict': subgraph_dict}
        if SWEEP_TARGET == "transformer":
            model = Transformer_MPNN(**config)
        elif SWEEP_TARGET == "fractal_egnn":
            config['subgraph_dict'] = {'mode': "fractal"}
            if config['model_type'] == 'fractal_egnn':
                model = Fractal_EGNN(**config)
            elif config['model_type'] == 'fractal_egnn_v2':
                model = Fractal_EGNN_v2(**config)
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join("../trained/", 'MNIST', 'Transformer'),
                                              filename='{epoch:02d}-{val_loss:.2f}', save_top_k=3, monitor='val_loss',
                                              mode='min')

        trainer = pl.Trainer(max_epochs=config['epochs'],
                             logger=wandb_logger,
                             gradient_clip_val=1.0,
                             accelerator='gpu',
                             devices=1,
                             callbacks=[checkpoint_callback])
        lightning_model = MNISTModel(model, **config)
        trainer.fit(lightning_model)

wandb.agent(sweep_id, function=train)