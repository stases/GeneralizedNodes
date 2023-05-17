import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchvision import datasets, transforms


# Define your PyTorch Lightning model
class MyModel(pl.LightningModule):
    def __init__(self, hidden_dim=128, learning_rate=1e-3, residual=False):
        super().__init__()
        self.save_hyperparameters()

        self.layer = nn.Linear(28 * 28, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 10)
        self.residual = residual
    def forward(self, x):
        x_0 = x
        x = x.view(x.size(0), -1)  # flatten the input
        x = F.relu(self.layer(x))
        x = self.classifier(x)
        if self.residual:
            x += x_0
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        # MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transform),
            batch_size=64, shuffle=True)
        return train_loader


# Define your sweep
sweep_config = {
    'method': 'bayes',  # set the search method to Bayesian Optimization
    'metric': {
        'name': 'train_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'hidden_dim': {
            'values': [32, 64, 128, 256, 512]
        },
        'learning_rate': {
            'min': 1e-6,
            'max': 1e-2
        },
        'residual': {
            'values': [True, False]
        }
    }
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep_config)


# Define sweep function
def train():
    # Initialize wandb logger
    wandb_logger = WandbLogger()
    with wandb.init() as run:
        config = run.config
    # Define the model with hyperparameters from wandb
        model = MyModel(hidden_dim=config.hidden_dim,
                        learning_rate=config.learning_rate)

        # Define the trainer and train the model
        trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)
        trainer.fit(model)


# Run the sweep
wandb.agent(sweep_id, function=train)
