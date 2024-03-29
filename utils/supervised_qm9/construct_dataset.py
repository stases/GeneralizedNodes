from utils.supervised_qm9.QM9_hypernode import QM9_Hypernodes
from torch_geometric.datasets import QM9
import argparse

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--path",
        default='./data/qm9/',
        type=str,
        help="Choose the path of the original raw QM9 dataset.",
    )

    parser.add_argument(
        "--mode",
        default="en_ee",
        type=str,
        help="Choose the Hypernode construct. mode",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    path = kwargs.pop("path")
    mode = kwargs.pop("mode")
    # Download QM9 if its not there yet
    dataset = QM9(path)
    hyper = QM9_Hypernodes(path, mode = mode)