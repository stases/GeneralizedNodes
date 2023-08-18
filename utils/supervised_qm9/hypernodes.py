import torch
import copy
from torch_geometric.utils import add_self_loops, degree, to_undirected

def sphere_sample(electron_pos):
    spherical_noise = torch.randn_like(electron_pos)
    norm = torch.sqrt(torch.sum(torch.pow(spherical_noise, 2), dim=1)).repeat(3,1).T
    return spherical_noise/norm

class Hypernode:
    def __init__(self, mol, atom_info, mode, plot_mode = False):
        self.original_mol = mol
        self.hypermol = copy.copy(mol)
        self.num_atoms = self.hypermol.x.shape[0]
        self.atom_list = atom_info['atom_list']
        self.num_atom_types = atom_info['num_atom_types']
        self.num_electron_types = atom_info['num_electron_types']
        self.device = self.hypermol.x.device
        self.mode = mode
        self.plot_mode = plot_mode
        self.added_electrons = False

    def get_atom_info(self, atom_idx):
        for atom_type_idx, atom_info in enumerate(self.atom_list):
            if self.original_mol.x[atom_idx][atom_type_idx] == 1:
                return atom_info

    def get_bond_order(self, edge_idx):
        weights = torch.Tensor([1, 2, 3, 4]).to(self.device)
        return int(torch.sum(self.original_mol.edge_attr[edge_idx] * weights[None, :].to(self.device)))

    def get_num_bonds(self, atom_idx):
        # Used to weigh single/double/triple bonds
        weights = torch.Tensor([1, 2, 3, 4]).to(self.device)
        row, col = self.original_mol.edge_index
        num_bonds = torch.sum(self.original_mol.edge_attr[row == atom_idx] * weights[None, :].to(self.device))
        num_unique_bonds = degree(row)[atom_idx]
        return int(num_bonds), int(num_unique_bonds)

    def get_steric_number(self, atom_idx):
        # atom_name, num_electrons = self.get_atom_info(self.original_mol.x[atom_idx], self.atom_list)
        atom_name, num_electrons = self.get_atom_info(atom_idx)
        num_bonds, num_unique_bonds = self.get_num_bonds(atom_idx)
        if atom_name != 'H':
            # Assuming that the molecules achieve octet
            num_lone_pairs = (num_electrons - num_bonds) / 2
            steric_number = num_lone_pairs + num_unique_bonds
        else:
            steric_number = 1
        return steric_number

    def add_electrons(self):
        assert not self.added_electrons, "Electrons are already added."
        # Initial modification
        # Goal of this is to get rid of unecessary info contained in mol.x
        # Also, goal is to append "zeros" to match the size that corresponds
        # to the size of num_atoms + num_electron_types
        self.hypermol.x = torch.clone(self.original_mol.x)
        self.hypermol.x = torch.cat((self.hypermol.x[..., :self.num_atom_types],
                                     torch.zeros(self.num_atoms, self.num_electron_types).to(self.device)),
                                    dim=1)

        # Get steric number of all atoms in a molecules
        self.atom_steric_numbers = torch.zeros(self.num_atoms).to(self.device)  # Will be required for edge constr.
        self.atom_num_bonds = torch.zeros(self.num_atoms).to(self.device)
        # Begin constructing nodes
        for atom_idx in range(self.num_atoms):
            num_bonds, _ = self.get_num_bonds(atom_idx)
            # Steric number will be used as a one-hot feature of the electron (type)
            steric_number = int(self.get_steric_number(atom_idx))  # WARNING! (.5 problem)
            self.atom_steric_numbers[atom_idx] = steric_number
            self.atom_num_bonds[atom_idx] = num_bonds
            electron_feature = torch.zeros(self.num_electron_types + self.num_atom_types).to(self.device)
            electron_feature[self.num_atom_types + steric_number - 1] = 1
            self.hypermol.x = torch.cat((self.hypermol.x,
                                         electron_feature.repeat(num_bonds, 1)), dim=0)
            # Note: Assuming .repeat(num_bonds) ignores unpaired electron pairs.
            # This is due to the fact that there are non-octet molecules in the dataset
            # and therefore it is hard to count the correct number of unpaired pairs.

        self.added_electrons = True

    def add_pos(self):
        assert self.added_electrons, "Add electrons first by using the .add_electrons() method."
        self.hypermol.pos = torch.clone(self.original_mol.pos)
        # Begin constructing nodes
        for atom_idx in range(self.num_atoms):
            num_bonds = int(self.atom_num_bonds[atom_idx])
            electron_pos = self.original_mol.pos[atom_idx].repeat(num_bonds, 1)
            if self.plot_mode:
                pos_std = torch.std(torch.diff(self.original_mol.pos, dim=0))
                electron_pos += sphere_sample(electron_pos) * pos_std / 10
            self.hypermol.pos = torch.cat((self.hypermol.pos,
                                           electron_pos), dim=0)

    def add_edges(self):
        assert self.added_electrons, "Add electrons first by using the .add_electrons() method."
        self.hypermol.edge_index = torch.clone(self.original_mol.edge_index)
        self.added_edges = 0
        # 1) non-hybr., 2) sp1, 3) sp2, 4) sp3 
        # NOTE: treating H single electron as non-hybr.
        # It has steric number = 1
        row, col = self.original_mol.edge_index
        new_rows, new_cols = [], []
        # Begin constructing edges
        # Appending electron <-> nucleus edge pairs
        for atom_idx in range(self.num_atoms):
            for electron_idx in range(int(self.atom_num_bonds[atom_idx])):
                new_rows.append(atom_idx)
                # new_cols.append(self.num_atoms+electron_tracker)
                new_cols.append(self.num_atoms +
                                torch.sum(self.atom_num_bonds[:atom_idx]) +
                                electron_idx)
                self.added_edges += 1

        # Preparation for electron <-> electron pairs      
        # The tracker is used to see which electrons were already used
        # Used pairs are utilized to count edges only once
        # Undirected graph is obtained later by using the to_undirected method
        hyperedge_tracker = torch.zeros(self.num_atoms)
        used_pairs = []
        # Appending electron -> electron pairs
        for edge_idx in range(len(row)):
            bond_order = self.get_bond_order(edge_idx)
            sender_atom, reciever_atom = row[edge_idx], col[edge_idx]
            # Get the right electron position in the index (atom offset + electrons up to that point)
            sender_electron = self.num_atoms + torch.sum(self.atom_num_bonds[:sender_atom])
            reciever_electron = self.num_atoms + torch.sum(self.atom_num_bonds[:reciever_atom])
            # Checking whether electron pairs of adjacent atoms were already used
            # This way we prevent double counting which could potentially lead to 
            # incorrect calculations in the hyperedge tracker
            if ([sender_atom, reciever_atom] not in used_pairs):
                for bond_idx in range(bond_order):
                    new_rows.append(sender_electron + hyperedge_tracker[sender_atom])
                    new_cols.append(reciever_electron + hyperedge_tracker[reciever_atom])
                    hyperedge_tracker[sender_atom] += 1
                    hyperedge_tracker[reciever_atom] += 1
                    self.added_edges += 1
                used_pairs.append([sender_atom, reciever_atom])
                used_pairs.append([reciever_atom, sender_atom])
        if self.mode == "en_ee":
            self.hypermol.edge_index = torch.stack([torch.Tensor(new_rows), torch.Tensor(new_cols)], dim=0).to(self.device).type(torch.int64)
        if self.mode == "en_ee_nn":
            # Now also including previous pairs that correspond to nucleus <--> nucleus bonda
            self.hypermol.edge_index = torch.stack([torch.cat((torch.Tensor(new_rows), row), dim=0),
                                                    torch.cat((torch.Tensor(new_cols), col), dim=0)],
                                                   dim=0).to(self.device).type(torch.int64)
        #self.hypermol.edge_index = to_undirected(new_edge_idx)

    def add_edge_attr(self):
        if self.mode == "en_ee":
            num_classes = 2
        if self.mode == "en_ee_nn":
            num_classes = 3
        new_edge_attr = []
        for edge_idx, edge in enumerate(self.hypermol.edge_index[0]):
            if edge_idx < torch.sum(self.atom_num_bonds):
                new_edge_attr.append(0)
            elif torch.sum(self.atom_num_bonds) <= edge_idx < self.added_edges:
                new_edge_attr.append(1)
            else:
                # Runs only if mode is "en_ee_nn"
                new_edge_attr.append(2)
        new_edge_attr = torch.Tensor(new_edge_attr)
        new_edge_attr = torch.nn.functional.one_hot(new_edge_attr.to(torch.int64), num_classes)
        self.hypermol.edge_attr = new_edge_attr

    def construct_hypernode(self):
        self.add_electrons()
        self.add_pos()
        self.add_edges()
        self.add_edge_attr()
        self.hypermol.edge_index, self.hypermol.edge_attr = to_undirected(self.hypermol.edge_index, self.hypermol.edge_attr, reduce="mean")
        return self.hypermol