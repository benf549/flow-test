
import shelve
from typing import *
from tqdm import tqdm
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

from utils.openfold_rigid_utils import Rigid
from utils.build_rotamers import compute_alignment_matrices, apply_transformation, extend_coordinates
from utils.constants import MAX_PEPTIDE_LENGTH, ideal_prot_aa_coords, aa_short_to_idx, hydrogen_extended_dataset_atom_order


@dataclass
class ModelOutput:
    pred_trans: torch.Tensor
    pred_rotmats: torch.Tensor


@dataclass
class BatchData:
    r_1: Optional[Rigid]
    res_mask: torch.Tensor
    diffuse_mask: torch.Tensor
    batch_size: int
    num_res: int
    device: torch.device
    res_indices: torch.Tensor

    r_t: Rigid = field(init=False)
    so3_t: torch.Tensor = field(init=False)
    r3_t: torch.Tensor = field(init=False)
    trans_self_conditioning: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'BatchData':
        self.device = device
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
            elif isinstance(v, Rigid):
                self.__dict__[k] = v.to(device)
        return self

    def _update_from_corrupt(self, r_t: Rigid, so3_t: torch.Tensor, r3_t: torch.Tensor) -> None:
        self.r_t = r_t
        self.so3_t = so3_t
        self.r3_t = r3_t
    
    def _set_self_condition(self, tsc: torch.Tensor) -> None:
        self.trans_self_conditioning = tsc
    
    @property
    def trans_1(self) -> torch.Tensor:
        if self.r_1 is None:
            raise AttributeError("r_1 is not set.")
        return self.r_1.get_trans()
    
    @property
    def trans_t(self) -> torch.Tensor:
        if not hasattr(self, 'r_t'):
            raise AttributeError("trans_t / r_t is not set, run _update_from_corrupt first.")
        return self.r_t.get_trans()

    @property
    def rotmats_1(self) -> torch.Tensor:
        if self.r_1 is None:
            raise AttributeError("r_1 is not set.")
        return self.r_1.get_rots().get_rot_mats()

    @property
    def rotmats_t(self) -> torch.Tensor:
        if not hasattr(self, 'r_t'):
            raise AttributeError("rotmats_t / r_t is not set, run _update_from_corrupt first.")
        return self.r_t.get_rots().get_rot_mats()


class UnclusteredProteinChainDataset(Dataset):
    """
    Dataset where every pdb_assembly-segment-chain is a separate index.
    """
    def __init__(self, dataset_shelve_path, metadata_shelve_path):

        self.pdb_code_to_complex_data = shelve.open(str(dataset_shelve_path), 'r', protocol=5)

        metadata = shelve.open(str(metadata_shelve_path), 'r', protocol=5)
        self.chain_key_to_index = metadata['chain_key_to_index']
        self.index_to_complex_size = metadata['index_to_complex_size']
        self.index_to_chain_key = metadata['index_to_chain_key']
        self.index_to_num_ligand_contacting_residues = metadata['index_to_num_ligand_contacting_residues']
        metadata.close()

    def __del__(self) -> None:
        if hasattr(self, 'pdb_code_to_complex_data'):
            self.pdb_code_to_complex_data.close()

    def __len__(self) -> int:
        return len(self.chain_key_to_index)

    def __getitem__(self, index: int) -> Tuple[dict, str]:
        # Take indexes unique to chain and return the complex data for that chain and the chain key.
        chain_key = self.index_to_chain_key[index]
        pdb_code = chain_key.split('-')[0]
        output_data = self.pdb_code_to_complex_data[pdb_code]
        return output_data, chain_key
    
    def write_all_sequence_fasta(self, output_path: str) -> None:
        """
        Writes all sequences longer than MAX_PEPTIDE_LENGTH to a fasta file.

        Run 30% cluster generation with:
            `mmseqs easy-cluster fasta.txt cluster30test tmp30test --min-seq-id 0.3 -c 0.5 --cov-mode 5 --cluster-mode 3`
        """
        output = {}
        # Loop over everything in the dataset.
        for pdb_code, data_dict in tqdm(self.pdb_code_to_complex_data.items(), total=len(self.pdb_code_to_complex_data)):
            for key, sub_data in data_dict.items():
                # Select the chains which are ('Segment', 'Chain') tuples and record crystallized sequence.
                if isinstance(key, tuple):
                    chain_key = "-".join([pdb_code, *key])
                    sequence = sub_data['polymer_seq']
                    output[chain_key] = sequence
        
        # Sort the output by chain_key so the fasta file is sorted.
        output = sorted(output.items(), key=lambda x: x[0])

        # Write the fasta file.
        with open(output_path, 'w') as f:
            for chain_key, sequence in output:
                if sequence is not None and len(sequence) > MAX_PEPTIDE_LENGTH:
                    f.write(f">{chain_key}\n")
                    f.write(f"{sequence}\n")


@torch.no_grad()
def idealize_backbone_coords(bb_coords, phi_psi_angles) -> torch.Tensor:
    # Expand copies of the idealized frames to match the number of frames in the batch
    ideal_ala_coords = ideal_prot_aa_coords[aa_short_to_idx['A']]
    ideal_frames_exp = ideal_ala_coords[[hydrogen_extended_dataset_atom_order['A'].index(x) for x in ('N', 'CA', 'C')]].to(bb_coords.device).unsqueeze(0).expand(bb_coords.shape[0], -1, -1)

    # Align the idealized frames from the origin to the actual backbone frames
    frame_alignment_matrices = compute_alignment_matrices(bb_coords[:, [0, 1, 3]], ideal_frames_exp)
    N, CA, C = apply_transformation(ideal_frames_exp, *frame_alignment_matrices).unbind(dim=1)

    # Compute virtual CB coordinates for idealized frames.
    b = CA - N
    c = C - CA
    a = torch.cross(b, c, dim=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

    # Compute the oxygen coordinates for the ideal frames (psi-dependent so need to do some math).
    dihedral_angles_rad = (phi_psi_angles[:, 1].unsqueeze(1) + 180.0).nan_to_num().deg2rad() # just use 0.0 deg for missing angles.
    bond_angles = torch.full_like(dihedral_angles_rad, fill_value=120.8).deg2rad() # Ca-C-O angle in radians
    bond_lengths = torch.full_like(dihedral_angles_rad, fill_value=1.23) # C-O bond length
    O = extend_coordinates(torch.stack([N, CA, C], dim=1), bond_lengths, bond_angles, dihedral_angles_rad)

    return torch.stack([N, CA, CB, C, O], dim=1)
