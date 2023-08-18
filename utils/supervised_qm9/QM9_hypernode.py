from torch_geometric.datasets import QM9
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
import sys
sys.path.append("../")
from utils.supervised_qm9.hypernodes import *


atom_info = {'atom_list': [('H', 1), ('C', 4), ('N', 5), ('O', 6), ('F', 7)],
             'num_atom_types': 5,
             'num_electron_types': 4}

class QM9_Hypernodes(InMemoryDataset):
    def __init__(self, root, mode, transform=None, pre_transform=None, pre_filter=None):
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root
    @property
    def raw_file_names(self):
        return ['data_v3.pt'];

    @property
    def processed_file_names(self):
        if self.mode == "en_ee":
            return ['data_v3_hypernodes_en_ee.pt']
        if self.mode == "en_ee_nn":
            return ['data_v3_hypernodes_en_ee_nn.pt']

    def download(self):
        # Download to `self.raw_dir`.
        #download_url(url, self.raw_dir)
        # TO BE IMPLEMENTED
        1 == 1

    def process(self):
        # Read data into huge `Data` list.
        data_list = QM9(self.root)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        hyper_data_list = []
        print('Processing dataset into hypernode dataset:')
        for mol_idx, mol in enumerate(tqdm(data_list)):
            hypermol = Hypernode(mol, atom_info, self.mode)
            hyper_data_list.append(hypermol.construct_hypernode())
        torch.save(self.collate(hyper_data_list), self.processed_paths[0])