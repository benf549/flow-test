
import shelve
from typing import *
from tqdm import tqdm
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import pandas as pd
from torch.utils.data import Dataset, Sampler

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


def invert_dict(d: dict) -> dict:
    clusters = defaultdict(list)
    for k, v in d.items():
        clusters[v].append(k)
    return dict(clusters)


def filter_subclusters(subclusters_info: dict, curr_clusters: dict) -> dict:
    """
    Filters out subclusters that are not in the current cluster set.
    """
    filtered_subclusters = defaultdict(lambda: defaultdict(list))
    for cluster, subcluster_dict in subclusters_info.items():
        if cluster in curr_clusters:
            for subcluster_centroid, subcluster_list in subcluster_dict.items():
                for subcluster_entry in subcluster_list:
                    if subcluster_entry in curr_clusters[cluster]:
                        filtered_subclusters[cluster][subcluster_centroid].append(subcluster_entry)
    
    return filtered_subclusters


class ClusteredDatasetSampler(Sampler):
    """
    Samples a single protein complex from precomputed mmseqs clusters.
    Ensures samples drawn evenly by sampling first from sequence clusters, then by pdb_code, then by assembly and chain.
    Iteration returns batched indices for use in UnclusteredProteinChainDataset.
    Pass to a DataLoader as a batch_sampler.
    """
    def __init__(
        self, dataset: UnclusteredProteinChainDataset, is_test_dataset_sampler: bool, 
        batch_size, sample_randomly, max_protein_size, clustering_dataframe_path,
        subcluster_pickle_path, debug, soluble_proteins_only: bool = False,
        single_protein_debug: bool = False, seed: Optional[int] = None, 
        subset_pdb_code_list: Optional[List[str]] = None, shuffle=True
    ):
        # Set the random seed for reproducibility and consistent randomness between processes if parallelized.
        if seed is None:
            self.generator = torch.Generator(device='cpu')
        else:
            self.generator = torch.Generator().manual_seed(seed)

        # The unclustered dataset where each complex/assembly is a single index.
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_protein_length = max_protein_size

        self.subset_pdb_code_set = set(subset_pdb_code_list) if subset_pdb_code_list is not None else {}
        if debug:
            self.subset_pdb_code_set = {'6w70'}

        # Load the cluster data.
        sequence_clusters = pd.read_pickle(clustering_dataframe_path)

        # if debug:
        #    sequence_clusters = sequence_clusters[sequence_clusters.chain.str.find('w7') == 1]

        if debug:
            train_cluster_meta = sequence_clusters
            test_cluster_meta = sequence_clusters
        else:
            train_cluster_meta = sequence_clusters[sequence_clusters.is_train & (~sequence_clusters.contaminated) & (~sequence_clusters.strep_structural_chain_contam) & (~sequence_clusters.strep_structural_bioa_contam)]
            test_cluster_meta = sequence_clusters[(~sequence_clusters.is_train) & (~sequence_clusters.contaminated) & (~sequence_clusters.strep_structural_chain_contam) & (~sequence_clusters.strep_structural_bioa_contam)]

        if soluble_proteins_only:
            if len(self.subset_pdb_code_set) > 0:
                raise NotImplementedError("Subset pdb codes not implemented with soluble proteins only.")
            possible_membrane_pdb_codes = set(pd.read_csv(str(Path(__file__).parent.parent / 'files/membrane_excluded_PDBs.csv'), index_col=0).PDB_IDS)

            if is_test_dataset_sampler:
                self.subset_pdb_code_set = set(test_cluster_meta.chain.str.slice(0, 4).to_list()) - possible_membrane_pdb_codes 
            else:
                self.subset_pdb_code_set = set(train_cluster_meta.chain.str.slice(0, 4).to_list()) - possible_membrane_pdb_codes

        # Maps sequence cluster to number of chains and vice versa
        self.chain_to_cluster = sequence_clusters.set_index('chain').to_dict()['cluster_representative']
        self.cluster_to_chains = invert_dict(self.chain_to_cluster)
        self.subclusters_info = pd.read_pickle(subcluster_pickle_path)
        
        # Load relevant pickled sets of cluster keys, filter for train/test as necessary.
        self.train_split_clusters = set(train_cluster_meta['cluster_representative'].unique())
        self.test_split_clusters = set(test_cluster_meta['cluster_representative'].unique())
        self.cluster_to_chains, self.subclusters_info = self.filter_clusters(is_test_dataset_sampler, single_protein_debug)

        # Sample the first epoch.
        self.curr_samples = []
        self.sample_clusters()

        self.curr_batches = []
        self.construct_batches()

    def __len__(self) -> int:
        """
        Returns number of batches in the current epoch.
        """
        return len(self.curr_batches)

    def filter_clusters(self, is_test_dataset_sampler: bool, single_protein_debug: bool) -> dict:
        """
        Filter clusters based on the given dataset sampler and the max protein length.
            Parameters:
            - is_test_dataset_sampler (bool): True if the dataset sampler is for the test dataset, False otherwise.

            Returns:
            - dict: A dictionary containing the filtered clusters.
        """

        if is_test_dataset_sampler:
            curr_cluster_set = self.test_split_clusters
        else:
            curr_cluster_set = self.train_split_clusters
        
        if single_protein_debug:
            output = {}
            n_copies = 1200 if not is_test_dataset_sampler else 100
            for idx in range(n_copies):
                output[f'debug_{idx}-A-A'] = ['6w70_1-A-A']
            return output

        if self.cluster_to_chains is None:
            raise NotImplementedError("Unreachable.")

        # Filter the clusters for train or test set.
        use_subset_pdb_codes = len(self.subset_pdb_code_set) > 0
        output = {k: v for k,v in self.cluster_to_chains.items() if k in curr_cluster_set}

        # If we don't have a max protein length, return the output.
        if self.max_protein_length is None:
            return output, filter_subclusters(self.subclusters_info, output)

        # Drop things that are longer than the max protein length and not in the subset pdb code set.
        filtered_output = defaultdict(list)
        for cluster_rep, cluster_list in output.items():
            for chain in cluster_list:
                if chain not in self.dataset.chain_key_to_index:
                    continue
                if (use_subset_pdb_codes and not (chain.split('_')[0] in self.subset_pdb_code_set)):
                    continue
                chain_len = self.dataset.index_to_complex_size[self.dataset.chain_key_to_index[chain]]
                if chain_len <= self.max_protein_length:
                    filtered_output[cluster_rep].append(chain)

        return filtered_output, filter_subclusters(self.subclusters_info, filtered_output)
    
    def sample_clusters(self) -> None:
        """
        Randomly samples clusters from the dataset for the next epoch.
        Updates the self.curr_samples list with new samples.
        """
        self.curr_samples = []
        # Loop over the clusters and sample a random chain from each.
        for cluster, subclusters in self.subclusters_info.items():
            # Get a random subcluster.
            subcluster_sample_index = int(torch.randint(0, len(subclusters), (1,), generator=self.generator).item())
            subcluster_key = list(subclusters.keys())[subcluster_sample_index]

            # Get a random chain from the subcluster.
            subcluster_element_sampler_index = int(torch.randint(0, len(subclusters[subcluster_key]), (1,), generator=self.generator).item())
            subcluster_element_key = subclusters[subcluster_key][subcluster_element_sampler_index]

            self.curr_samples.append(self.dataset.chain_key_to_index[subcluster_element_key])

    def construct_batches(self):
        """
        Batches by size inspired by:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler:~:text=%3E%3E%3E%20class%20AccedingSequenceLengthBatchSampler
        """
        # Reset the current batches.
        self.curr_batches = []

        # Sort the samples by size.
        curr_samples_tensor = torch.tensor(self.curr_samples)
        sizes = torch.tensor([self.dataset.index_to_complex_size[x] for x in self.curr_samples])
        size_sort_indices = torch.argsort(sizes)

        # iterate through the samples in order of size, create batches of size batch_size.
        debug_sizes = []
        curr_list_sample_indices, curr_list_sizes = [], []
        for curr_size_sort_index in size_sort_indices:
            # Get current sample index and size.
            curr_sample_index = curr_samples_tensor[curr_size_sort_index].item()
            curr_size = sizes[curr_size_sort_index].item()

            # Add to the current batch if would not exceed batch size otherwise create a new batch.
            if sum(curr_list_sizes) + curr_size <= self.batch_size:
                curr_list_sample_indices.append(curr_sample_index)
                curr_list_sizes.append(curr_size)
            else:
                # Add the current batch to the list of batches.
                self.curr_batches.append(curr_list_sample_indices)
                debug_sizes.append(sum(curr_list_sizes))

                # Reset the current batch.
                curr_list_sizes = [curr_size]
                curr_list_sample_indices = [curr_sample_index]

        # Store any remaining samples.
        if len(curr_list_sample_indices) > 0:
            self.curr_batches.append(curr_list_sample_indices)
            debug_sizes.append(sum(curr_list_sizes))

        # Shuffle the batches.
        if self.shuffle:
            shuffle_indices = torch.randperm(len(self.curr_batches), generator=self.generator).tolist()
            curr_batches_ = [self.curr_batches[x] for x in shuffle_indices]
            self.curr_batches = curr_batches_

        # Sanity check that we have the correct number of samples after iteration.
        assert sum(debug_sizes) == sizes.sum().item(), "Mismatch between number of samples and expected size of samples."

    def __iter__(self):
        # Yield the batches we created.
        for batch in self.curr_batches:
            yield batch

        # Resample for the next epoch, and create new batches.
        self.sample_clusters()
        self.construct_batches()

