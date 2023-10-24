import numpy as np
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.datasets.wikics import WikiCS
from torch_geometric.loader import DataLoader
from models.gnn.networks import *
from utils.transforms import Graph_to_Subgraph, Fully_Connected_Graph

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


def calculate_accuracy(output, labels, mask):
    _, predicted = torch.max(output[mask], dim=1)
    correct = (predicted == labels[mask]).sum().item()
    return correct / mask.sum().item()


def train_wikics_split(model, epochs, mask_idx, dataset, batch_size, optimizer, scheduler, criterion, subgraph_flag, verbose=False, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_mask = dataset[0].train_mask[:, mask_idx]
    val_mask = dataset[0].val_mask[:, mask_idx]
    test_mask = dataset[0].test_mask
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    # Best validation accuracy and corresponding epoch
    best_val_acc = 0.0
    best_epoch = 0

    # Counter for early stopping
    no_improvement_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Training phase
        for batch in loader:
            batch = batch.to("cuda")
            # Check if the model and the data are on GPU
            optimizer.zero_grad()
            output = model(batch)
            y = batch.y
            if subgraph_flag:
                output = output[batch.ground_node]
            loss = criterion(output[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        # Calculate training accuracy
        train_acc = calculate_accuracy(output, y, train_mask)
        if verbose:
            print(f'Epoch: {epoch:03d}, Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}')
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in loader:
                batch = batch.to(device)
                output = model(batch)
                y = batch.y
                if subgraph_flag:
                    output = output[batch.ground_node]
                val_loss += criterion(output[val_mask], y[val_mask]).item()

            # Calculate validation accuracy
            val_acc = calculate_accuracy(output, y, val_mask)
            if verbose:
                print(f'Epoch: {epoch:03d}, Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        # Save the model with the best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # Early stopping
        if no_improvement_counter >= 30:
            if verbose:
                print(
                    f'No improvement in validation accuracy for 30 epochs. Best validation accuracy is {best_val_acc} at epoch {best_epoch}.')
            break

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Testing phase
    with torch.no_grad():
        test_loss = 0.0
        for batch in loader:
            batch = batch.to("cuda")
            y = batch.y


            output = model(batch)
            if subgraph_flag:
                output = output[batch.ground_node]
            test_loss += criterion(output[test_mask], y[test_mask]).item()

        # Calculate test accuracy
        test_acc = calculate_accuracy(output, y, test_mask)

    print(f'Best epoch: {best_epoch}')
    print(f'Test Accuracy: {test_acc:.4f} on split {mask_idx}')
    print(f'Best Validation Accuracy: {best_val_acc:.4f} on split {mask_idx}')
    return {'mask_idx': mask_idx, 'test_acc': test_acc, 'best_epoch': best_epoch, 'best_val_acc': best_val_acc, 'train_acc': train_acc}


def train_wikics_model(data_dir, model_arch, learning_rate, epochs, batch_size, transform, subgraph_dict, **kwargs):
    wandb.init(project="wikics", config=kwargs)
    # Set up the learning rate scheduler
    train_acc = []
    val_acc = []
    test_acc = []
    if subgraph_dict is None:
        subgraph_flag = False
        transform = None
    else:
        subgraph_flag = True
        mode = subgraph_dict['mode']
        transform = Graph_to_Subgraph(mode=mode)
        print("Using subgraph mode: ", mode)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for mask_idx in range(20):
        model_class = MODEL_MAP.get(model_arch, None)
        #print("Using model: ", model_class)
        if model_class is None:
            raise ValueError(f"Invalid model_arch value: {model_arch}")
        # Instantiate the model using kwargs from the YAML configuration file
        if "model" in kwargs:
            kwargs.pop("model")
        model = model_class(**kwargs)
        model = model.to("cuda")
        # Check if the model is on the GPU
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        print('learning rate is', learning_rate)
        dataset = WikiCS(root=data_dir, transform=transform)
        results = train_wikics_split(model, epochs, mask_idx, dataset, batch_size, optimizer, scheduler, criterion, subgraph_flag, **kwargs)
        wandb.log(results)
        train_acc.append(results['train_acc'])
        val_acc.append(results['best_val_acc'])
        test_acc.append(results['test_acc'])

    # Calculate means
    train_acc_mean = np.mean(train_acc)
    val_acc_mean = np.mean(val_acc)
    test_acc_mean = np.mean(test_acc)

    # Calculate standard deviations
    train_acc_std = np.std(train_acc)
    val_acc_std = np.std(val_acc)
    test_acc_std = np.std(test_acc)

    final_results = {
        "train_acc_mean": train_acc_mean,
        "val_acc_mean": val_acc_mean,
        "test_acc_mean": test_acc_mean,
        "train_acc_std": train_acc_std,
        "val_acc_std": val_acc_std,
        "test_acc_std": test_acc_std,
    }

    # Log final results
    wandb.log(final_results)