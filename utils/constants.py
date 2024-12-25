import os
import torch
import torch.nn.functional as F
from pathlib import Path

parent_path = Path(__file__).parent.absolute()

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE


# Define distance that will be used to create a mask for residues.
MIN_TM_SCORE_FOR_SIMILARITY = 0.70
MAX_PEPTIDE_LENGTH = 40
NUM_CB_ATOMS_FOR_BURIAL = 16
CHI_BIN_MIN, CHI_BIN_MAX = -180, 180
HARD_CLASH_TOLERANCE = 0.2
ANGSTROM_PER_BOHR = 0.5291772
HEAVY_ATOM_CONTACT_DISTANCE_THRESHOLD = 5.0

ON_ON_HYDROGEN_BOND_DISTANCE_CUTOFF = 3.5
ON_S_HYDROGEN_BOND_DISTANCE_CUTOFF = 4.2
S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF = 4.5
MIN_HBOND_ANGLE = 110
COVALENT_HYDROGEN_BOND_MAX_DISTANCE = 1.4
GENERIC_COVALENT_BOND_MAX_DISTANCE = 1.5

POSSIBLE_HYBRIDIZATION_LIST = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc']
POSSIBLE_FORMAL_CHARGE_LIST = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc']
POSSIBLE_NUM_HYDROGENS_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc']
POSSIBLE_DEGREE_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc']
POSSIBLE_IS_AROMATIC_LIST = [True, False]

# Map of canonical amino acid 1 to 3 letter codes.
aa_short_to_long = {'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS', 'I': 'ILE', 'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'N': 'ASN', 'G': 'GLY', 'H': 'HIS', 'L': 'LEU', 'R': 'ARG', 'W': 'TRP', 'A': 'ALA', 'V': 'VAL', 'E': 'GLU', 'Y': 'TYR', 'M': 'MET', 'X': 'XAA'}

aa_long_to_short = {x: y for y, x in aa_short_to_long.items()}
aa_long_to_idx = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19, 'XAA': 20}
aa_short_to_idx = {x: aa_long_to_idx[y] for x, y in aa_short_to_long.items()}
aa_idx_to_long = {x: y for y, x in aa_long_to_idx.items()}
aa_idx_to_short = {x: aa_long_to_short[y] for x, y in aa_idx_to_long.items()}

atomic_number_to_atom = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'}
atom_to_atomic_number = {v: k for k, v in atomic_number_to_atom.items()}
atom_to_atomic_number.update({'D': 1})
atom_to_period_idx = {'H': 0, 'He': 0, 'Li': 1, 'Be': 1, 'B': 1, 'C': 1, 'N': 1, 'O': 1, 'F': 1, 'Ne': 1, 'Na': 2, 'Mg': 2, 'Al': 2, 'Si': 2, 'P': 2, 'S': 2, 'Cl': 2, 'Ar': 2, 'K': 3, 'Ca': 3, 'Sc': 3, 'Ti': 3, 'V': 3, 'Cr': 3, 'Mn': 3, 'Fe': 3, 'Co': 3, 'Ni': 3, 'Cu': 3, 'Zn': 3, 'Ga': 3, 'Ge': 3, 'As': 3, 'Se': 3, 'Br': 3, 'Kr': 3, 'Rb': 4, 'Sr': 4, 'Y': 4, 'Zr': 4, 'Nb': 4, 'Mo': 4, 'Tc': 4, 'Ru': 4, 'Rh': 4, 'Pd': 4, 'Ag': 4, 'Cd': 4, 'In': 4, 'Sn': 4, 'Sb': 4, 'Te': 4, 'I': 4, 'Xe': 4, 'Cs': 5, 'Ba': 5, 'La': 5, 'Ce': 5, 'Pr': 5, 'Nd': 5, 'Pm': 5, 'Sm': 5, 'Eu': 5, 'Gd': 5, 'Tb': 5, 'Dy': 5, 'Ho': 5, 'Er': 5, 'Tm': 5, 'Yb': 5, 'Lu': 5, 'Hf': 5, 'Ta': 5, 'W': 5, 'Re': 5, 'Os': 5, 'Ir': 5, 'Pt': 5, 'Au': 5, 'Hg': 5, 'Tl': 5, 'Pb': 5, 'Bi': 5, 'Po': 5, 'At': 5, 'Rn': 5, 'Fr': 6, 'Ra': 6, 'Ac': 6, 'Th': 6, 'Pa': 6, 'U': 6, 'Np': 6, 'Pu': 6, 'Am': 6, 'Cm': 6, 'Bk': 6, 'Cf': 6, 'Es': 6, 'Fm': 6, 'Md': 6, 'No': 6, 'Lr': 6, 'Rf': 6, 'Db': 6, 'Sg': 6, 'Bh': 6, 'Hs': 6, 'Mt': 6, 'Ds': 6, 'Rg': 6, 'Cn': 6, 'Nh': 6, 'Fl': 6, 'Mc': 6, 'Lv': 6, 'Ts': 6, 'Og': 6}
atom_to_group_idx = {'H': 0, 'He': 17, 'Li': 0, 'Be': 1, 'B': 12, 'C': 13, 'N': 14, 'O': 15, 'F': 16, 'Ne': 17, 'Na': 0, 'Mg': 1, 'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'Ar': 17, 'K': 0, 'Ca': 1, 'Sc': 2, 'Ti': 3, 'V': 4, 'Cr': 5, 'Mn': 6, 'Fe': 7, 'Co': 8, 'Ni': 9, 'Cu': 10, 'Zn': 11, 'Ga': 12, 'Ge': 13, 'As': 14, 'Se': 15, 'Br': 16, 'Kr': 17, 'Rb': 0, 'Sr': 1, 'Y': 2, 'Zr': 3, 'Nb': 4, 'Mo': 5, 'Tc': 6, 'Ru': 7, 'Rh': 8, 'Pd': 9, 'Ag': 10, 'Cd': 11, 'In': 12, 'Sn': 13, 'Sb': 14, 'Te': 15, 'I': 16, 'Xe': 17, 'Cs': 0, 'Ba': 1, 'La': 2, 'Ce': torch.nan, 'Pr': torch.nan, 'Nd': torch.nan, 'Pm': torch.nan, 'Sm': torch.nan, 'Eu': torch.nan, 'Gd': torch.nan, 'Tb': torch.nan, 'Dy': torch.nan, 'Ho': torch.nan, 'Er': torch.nan, 'Tm': torch.nan, 'Yb': torch.nan, 'Lu': torch.nan, 'Hf': 3, 'Ta': 4, 'W': 5, 'Re': 6, 'Os': 7, 'Ir': 8, 'Pt': 9, 'Au': 10, 'Hg': 11, 'Tl': 12, 'Pb': 13, 'Bi': 14, 'Po': 15, 'At': 16, 'Rn': 17, 'Fr': 0, 'Ra': 1, 'Ac': 2, 'Th': torch.nan, 'Pa': torch.nan, 'U': torch.nan, 'Np': torch.nan, 'Pu': torch.nan, 'Am': torch.nan, 'Cm': torch.nan, 'Bk': torch.nan, 'Cf': torch.nan, 'Es': torch.nan, 'Fm': torch.nan, 'Md': torch.nan, 'No': torch.nan, 'Lr': torch.nan, 'Rf': 3, 'Db': 4, 'Sg': 5, 'Bh': 6, 'Hs': 7, 'Mt': 8, 'Ds': 9, 'Rg': 10, 'Cn': 11, 'Nh': 12, 'Fl': 13, 'Mc': 14, 'Lv': 15, 'Ts': 16, 'Og': 17}

period_group_tuple_to_atom_symbol = {}
for symbol, period in atom_to_period_idx.items():
    period_group_tuple_to_atom_symbol[(period, atom_to_group_idx[symbol])] = symbol

# Map of one letter amino acid codes to their corresponding atom order.
dataset_atom_order = {
    'G': ['N', 'CA', 'C', 'O'],
    'X': ['N', 'CA', 'C', 'O'],
    'A': ['N', 'CA', 'C', 'O', 'CB'],
    'S': ['N', 'CA', 'C', 'O', 'CB', 'OG', 'HG'],
    'C': ['N', 'CA', 'C', 'O', 'CB', 'SG', 'HG'],
    'T': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', 'HG1'],
    'P': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    'V': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
    'M': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    'N': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    'I': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    'L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    'E': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'K': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'Q': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'H': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', 'HD1', 'HE2'],
    'F': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'R': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'Y': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'HH'],
    'W': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2']
}

# full_atom_coords = N x 14 x 3 tensor of coordinates of all atoms in a protein.
# residue_metadata = N x 1  tensor 
# assert full_atom_coords.shape[0] == residue_metadata.shape[0], "Shapes must be the same"

rotatable_hydrogens = {('SER', 'HG'), ('CYS',  'HG'), ('THR', 'HG1'), ('TYR', 'HH')}
optional_hydrogen_map = {
    'C': ['HG'],
    'H': ['HD1', 'HE2'],
}

# The maximum number of atoms for a single residue.
MAX_NUM_RESIDUE_ATOMS = max([len(res) for res in dataset_atom_order.values()])

# Map from chi angle index to the atoms that define it. Includes rotatable hydrogen chi angles.
aa_to_chi_angle_atom_map = {
    'C': {1: ('N', 'CA', 'CB', 'SG'), 2: ('CA', 'CB', 'SG', 'HG')},
    'D': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'OD1')},
    'E': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'OE1')},
    'F': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1')},
    'H': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'ND1')},
    'I': {1: ('N', 'CA', 'CB', 'CG1'), 2: ('CA', 'CB', 'CG1', 'CD1')},
    'K': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'CE'), 4: ('CG', 'CD', 'CE', 'NZ')},
    'L': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1')},
    'M': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'SD'), 3: ('CB', 'CG', 'SD', 'CE')},
    'N': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'OD1')},
    'P': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD')},
    'Q': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'OE1')},
    'R': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'NE'), 4: ('CG', 'CD', 'NE', 'CZ')},
    'S': {1: ('N', 'CA', 'CB', 'OG'), 2: ('CA', 'CB', 'OG', 'HG')},
    'T': {1: ('N', 'CA', 'CB', 'OG1'), 2: ('CA', 'CB', 'OG1', 'HG1')},
    'V': {1: ('N', 'CA', 'CB', 'CG1')},
    'W': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1')},
    # NOTE: need to align leftover atoms to the first two chi angles before the final chi angle only for TYR.
    'Y': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1'), 3: ('CE1', 'CZ', 'OH', 'HH')}
}

##### Converts the atom names defined above to indices in the dataset atom order tensor
#### Assumes tensors are padded with with NaN at MAX_NUM_RESIDUE_ATOMS'th index in dim 1 
placeholder_indices = torch.tensor([MAX_NUM_RESIDUE_ATOMS] * 4)
aa_to_chi_angle_atom_index = torch.full((20, 4, 4), MAX_NUM_RESIDUE_ATOMS)
aa_to_chi_angle_mask = torch.full((21, 4), False)
aa_to_leftover_atoms = torch.full((20, MAX_NUM_RESIDUE_ATOMS), MAX_NUM_RESIDUE_ATOMS)
# Iterate in the order of the canonical amino acid indices in aa_idx_to_short
for idx in range(21):
    if idx == 20:
        aa = 'G'
    else:
        aa = aa_idx_to_short[idx]
    if aa in aa_to_chi_angle_atom_map:
        # Fill residues that have chi angles with indices of relevant atoms.
        all_atoms_set = set([x for x in range(len(dataset_atom_order[aa]))])
        chi_placed_atoms_set = set()
        for chi_num, atom_names in aa_to_chi_angle_atom_map[aa].items():
            chi_placed_indices = [dataset_atom_order[aa].index(x) for x in atom_names]
            aa_to_chi_angle_atom_index[idx, chi_num - 1] = torch.tensor(chi_placed_indices)
            chi_placed_atoms_set.update(chi_placed_indices)

        # Track which atoms are not involved in the placement process
        leftovers = sorted(list(all_atoms_set - chi_placed_atoms_set - {2, 3}))
        aa_to_leftover_atoms[idx, :len(leftovers)] = torch.tensor(leftovers)
        
        # Fill mask with True for chi angles and False for padding.
        aa_to_chi_angle_mask[idx, :len(aa_to_chi_angle_atom_map[aa])] = True

# Remove the terminal TYR chi angle atoms that we won't actually place when adjusting angles.
tyr_idx = aa_short_to_idx['Y']
num_tyr_leftover = (aa_to_leftover_atoms[tyr_idx] != MAX_NUM_RESIDUE_ATOMS).sum().item() # type: ignore
tyr_leftover_updated = sorted(list(set(aa_to_leftover_atoms[tyr_idx, :num_tyr_leftover].tolist() + [dataset_atom_order['Y'].index(y) for y in ['CE1', 'CZ', 'OH', 'HH']]))) # type: ignore
aa_to_leftover_atoms[tyr_idx, :len(tyr_leftover_updated)] = torch.tensor(tyr_leftover_updated)
aa_to_leftover_atoms = aa_to_leftover_atoms.narrow(dim=1, start=0, length=(aa_to_leftover_atoms != 14).sum(dim=1).max()) # type: ignore

# Load precomputed ideal coordinates, bond lengths, and bond angles necessary to build rotamers. 
# Computed from idealized single amino acid PDB files generated by something like Rosetta/MD/DFT.
ideal_aa_coords = torch.load(os.path.join(parent_path, '../files', 'new_ideal_coords.pt'))
ideal_bond_lengths = torch.load(os.path.join(parent_path, '../files', 'new_ideal_bond_lengths.pt'))
ideal_bond_angles = torch.load(os.path.join(parent_path, '../files', 'new_ideal_bond_angles.pt'))
alignment_indices = torch.load(os.path.join(parent_path, '../files', 'rotamer_alignment.pt'))
alignment_indices_mask = alignment_indices == -1
alignment_indices[alignment_indices_mask] = MAX_NUM_RESIDUE_ATOMS

# Heavy atoms that are used to align hydrogens from protonated idealized amino acids.
# Handles all hydrogens except backbone amide hydrogens.
hydrogen_alignment_coord_map = {
    'G': {('N', 'CA', 'C'): ['HA2', 'HA3']},
    'A': {('N', 'CA', 'CB'): ['HA', 'HB1', 'HB2', 'HB3']},
    'S': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'OG'): ['HB2', 'HB3']},
    'C': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'SG'): ['HB2', 'HB3']},
    'T': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG2'): ['HB', 'HG21', 'HG22', 'HG23']},
    'P': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG'): ['HB2', 'HB3'], ('CB', 'CG', 'CD'): ['HG2', 'HG3'], ('CG', 'CD', 'N'): ['HD2', 'HD3']},
    'V': {('N', 'CA', 'CB'): ['HA'], ('CG2', 'CB', 'CG1'): ['HB', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23']},
    'M': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG'): ['HB2', 'HB3'], ('CB', 'CG', 'SD'): ['HG2', 'HG3'], ('CG', 'SD', 'CE'): ['HE1', 'HE2', 'HE3']},
    'N': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG'): ['HB2', 'HB3'], ('OD1', 'CG', 'ND2'): ['HD21', 'HD22']},
    'I': {('N', 'CA', 'CB'): ['HA'], ('CG1', 'CB', 'CG2'): ['HB', 'HG21', 'HG22', 'HG23'], ('CB', 'CG1', 'CD1'): ['HG12', 'HG13', 'HD11', 'HD12', 'HD13']},
    'L': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG'): ['HB2', 'HB3'], ('CD1', 'CG', 'CD2'): ['HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23']},
    'D': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG'): ['HB2', 'HB3']},
    'E': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG'): ['HB2', 'HB3'], ('CB', 'CG', 'CD'): ['HG2', 'HG3']},
    'K': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG'): ['HB2', 'HB3'], ('CB', 'CG', 'CD'): ['HG2', 'HG3'], ('CG', 'CD', 'CE'): ['HD2', 'HD3'], ('CD', 'CE', 'NZ'): ['HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3']},
    'Q': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG'): ['HB2', 'HB3'], ('CB', 'CG', 'CD'): ['HG2', 'HG3'], ('OE1', 'CD', 'NE2'): ['HE21', 'HE22']},
    'H': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG'): ['HB2', 'HB3'], ('CE1', 'NE2', 'CD2'): ['HE1', 'HD2']},
    'F': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG'): ['HB2', 'HB3'], ('CE1', 'CZ', 'CE2'): ['HD1', 'HD2', 'HE1', 'HE2', 'HZ']},
    'R': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG'): ['HB2', 'HB3'], ('CB', 'CG', 'CD'): ['HG2', 'HG3'], ('CG', 'CD', 'NE'): ['HD2', 'HD3'], ('NE', 'CZ', 'NH2'): ['HE', 'HH21', 'HH22', 'HH12', 'HH11']},
    'Y': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG'): ['HB2', 'HB3'], ('CE1', 'CZ', 'CE2'): ['HE1', 'HE2', 'HD1', 'HD2']},
    'W': {('N', 'CA', 'CB'): ['HA'], ('CA', 'CB', 'CG'): ['HB2', 'HB3'], ('CD1', 'CG', 'CD2'): ['HD1', 'HE1', 'HE3', 'HZ2', 'HZ3', 'HH2']}
}

hydrogen_extended_dataset_atom_order = {
    'G': dataset_atom_order['G'] + ['HA2', 'HA3'],
    'A': (x_placeholder := dataset_atom_order['A'] + ['HA', 'HB1', 'HB2', 'HB3']),
    'S': dataset_atom_order['S'] + ['HA', 'HB2', 'HB3'],
    'C': dataset_atom_order['C'] + ['HA', 'HB2', 'HB3'],
    'T': dataset_atom_order['T'] + ['HA', 'HB', 'HG21', 'HG22', 'HG23'],
    'P': dataset_atom_order['P'] + ['HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3'],
    'V': dataset_atom_order['V'] + ['HA', 'HB', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23'],
    'M': dataset_atom_order['M'] + ['HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HE1', 'HE2', 'HE3'],
    'N': dataset_atom_order['N'] + ['HA', 'HB2', 'HB3', 'HD21', 'HD22'],
    'I': dataset_atom_order['I'] + ['HA', 'HB', 'HG21', 'HG22', 'HG23', 'HD11', 'HD12', 'HD13', 'HG12', 'HG13'],
    'L': dataset_atom_order['L'] + ['HA', 'HB2', 'HB3', 'HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23'],
    'D': dataset_atom_order['D'] + ['HA', 'HB2', 'HB3'],
    'E': dataset_atom_order['E'] + ['HA', 'HB2', 'HB3', 'HG2', 'HG3'],
    'K': dataset_atom_order['K'] + ['HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3'],
    'Q': dataset_atom_order['Q'] + ['HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HE21', 'HE22'],
    'H': dataset_atom_order['H'] + ['HA', 'HB2', 'HB3', 'HE1', 'HD2'],
    'F': dataset_atom_order['F'] + ['HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ'],
    'R': dataset_atom_order['R'] + ['HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE', 'HH21', 'HH22', 'HH12', 'HH11'],
    'Y': dataset_atom_order['Y'] + ['HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2'],
    'W': dataset_atom_order['W'] + ['HA', 'HB2', 'HB3', 'HD1', 'HE1', 'HE3', 'HZ2', 'HZ3', 'HH2'],
    'X': x_placeholder
}

aa_to_sc_hbond_donor_to_heavy_atom = {
    'C': {'SG': ['HG']},
    'H': {'ND1': ['HD1'], 'NE2': ['HE2']}, # TODO: Update this to handle HIS protonation states.
    'K': {'NZ': ['HZ1', 'HZ2', 'HZ3']},
    'N': {'ND2': ['HD21', 'HD22']},
    'Q': {'NE2': ['HE21', 'HE22']},
    'R': {
        'NE': ['HE'],
        'NH1': ['HH11', 'HH12'],
        'NH2': ['HH21', 'HH22']
    },
    'S': {'OG': ['HG']},
    'T': {'OG1': ['HG1']},
    'W': {'NE1': ['HE1']},
    'Y': {'OH': ['HH']},
}

aa_to_hbond_donor_index_map = {}
for aa in aa_to_sc_hbond_donor_to_heavy_atom:
    curr_fn = hydrogen_extended_dataset_atom_order[aa].index
    aa_map = {curr_fn(x): [curr_fn(j) for j in y] for x,y in aa_to_sc_hbond_donor_to_heavy_atom[aa].items()}
    aa_to_hbond_donor_index_map[aa] = aa_map

aa_to_sc_hbond_acceptor_heavy_atom = {
    'G': ['O'],
    'A': ['O'],
    'S': ['O', 'OG'],
    'C': ['O', 'SG'],
    'T': ['O', 'OG1'],
    'P': ['O'],
    'V': ['O'],
    'M': ['O', 'SD'],
    'N': ['O', 'OD1'],
    'I': ['O'],
    'L': ['O'],
    'D': ['O', 'OD1', 'OD2'],
    'E': ['O', 'OE1', 'OE2'],
    'K': ['O'],
    'Q': ['O', 'OE1'],
    'H': ['O'],
    'F': ['O'],
    'R': ['O'],
    'Y': ['O', 'OH'],
    'W': ['O']
}

aa_to_hbond_acceptor_map = {}
for aa in aa_to_sc_hbond_acceptor_heavy_atom:
    curr_fn = hydrogen_extended_dataset_atom_order[aa].index
    aa_map = [curr_fn(x) for x in aa_to_sc_hbond_acceptor_heavy_atom[aa]]
    aa_to_hbond_acceptor_map[aa] = aa_map

MAX_NUM_PROTONATED_RESIDUE_ATOMS = max([len(x) for x in hydrogen_extended_dataset_atom_order.values()]) # type: ignore
MAX_NUM_TRIPLETS_PER_RESIDUE = max([len(x) for x in hydrogen_alignment_coord_map.values()])
MAX_NUM_HYDROGENS_ALIGNED_PER_TRIAD = max([max([len(y) for y in x.values()]) for x in hydrogen_alignment_coord_map.values()])
ideal_prot_aa_coords = torch.load(os.path.join(parent_path, '../files', 'ideal_aa_coords_prot.pt'))

### Need to map from indices of heavy atoms in built coords to indices of heavy atoms in ideal coords
####   that we can use to align nonpolar hydrogens.
# There are a max of 5 triplets of heavy atoms we need to align to place hydrogens for each residue.
aa_to_hydrogen_alignment_triad_indices = torch.full((20, MAX_NUM_TRIPLETS_PER_RESIDUE, 3), MAX_NUM_PROTONATED_RESIDUE_ATOMS)
# The indices of the hydrogens we need to align for each triplet of heavy atoms.
aa_to_hydrogen_alignment_index = torch.full((20, MAX_NUM_TRIPLETS_PER_RESIDUE, MAX_NUM_HYDROGENS_ALIGNED_PER_TRIAD), MAX_NUM_PROTONATED_RESIDUE_ATOMS)

for aa_idx in range(20):
    aa = aa_idx_to_short[aa_idx]
    for triad_idx, triad in enumerate(hydrogen_alignment_coord_map[aa]):
        # Get the indices of the heavy atoms we need to align.
        triad_indices = [hydrogen_extended_dataset_atom_order[aa].index(x) for x in triad]
        aa_to_hydrogen_alignment_triad_indices[aa_idx, triad_idx] = torch.tensor(triad_indices)
        # Get the indices of the hydrogens we need to align.
        hydrogen_indices = [hydrogen_extended_dataset_atom_order[aa].index(x) for x in hydrogen_alignment_coord_map[aa][triad]]
        aa_to_hydrogen_alignment_index[aa_idx, triad_idx, :len(hydrogen_indices)] = torch.tensor(hydrogen_indices)

sequence_index_to_atomic_numbers = torch.zeros(21, MAX_NUM_PROTONATED_RESIDUE_ATOMS + 1, dtype=torch.long)
for x,y in hydrogen_extended_dataset_atom_order.items():
    sequence_index_to_atomic_numbers[aa_short_to_idx[x]] = F.pad(torch.tensor([atom_to_atomic_number[z[0]] for z in y]), (0, MAX_NUM_PROTONATED_RESIDUE_ATOMS + 1 - len(y)), 'constant', -1)
