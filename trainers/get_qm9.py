from torch_geometric.datasets import QM9
import torch
from warnings import warn

def get_qm9_statistics(data_dir):
    # Get the mean and std of the QM9 dataset for each label.
    dataset = QM9(data_dir)
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    return mean, std

def get_qm9(data_dir, device="cuda", LABEL_INDEX = 7, transform=None):
    """Download the QM9 dataset from pytorch geometric. Put it onto the device. Split it up into trainers / validation / test.
    Args:
        data_dir: the directory to store the data.
        device: put the data onto this device.
    Returns:
        trainers dataset, validation dataset, test dataset.
    """
    dataset = QM9(data_dir, transform=transform)

    #TODO: Check if we need permutations
    '''# Permute the dataset
    try:
        permu = torch.load("permute.pt")
        dataset = dataset[permu]
    except FileNotFoundError:
        warn("Using non-standard permutation since permute.pt does not exist.")
        dataset, _ = dataset.shuffle(return_perm=True)'''

    # z score / standard score targets to mean = 0 and std = 1.
    # TODO: Ignore for now, done in the training loop
    '''mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, LABEL_INDEX].item(), std[:, LABEL_INDEX].item()
    '''
    # Move the data to the device (it should fit on lisa gpus)
    dataset.data = dataset.data.to(device)

    len_train = 100_000
    len_val = 10_000

    train = dataset[:len_train]
    valid = dataset[len_train : len_train + len_val]
    test = dataset[len_train + len_val :]

    assert len(dataset) == len(train) + len(valid) + len(test)

    return train, valid, test

# Write a function that rescales the values back to the original scale.
def rescale(y, mean, std):
    device = y.device
    mean = mean.to(device)
    std = std.to(device)
    return y * std + mean

# Write a function that returns the mean and std of the target values.
def get_mean_std(data_dir, device="cuda", LABEL_INDEX = 7, transform=None):
    dataset = QM9(data_dir, transform=transform)
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    return mean[:, LABEL_INDEX].item(), std[:, LABEL_INDEX].item()