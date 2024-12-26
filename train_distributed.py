import os
import io
import shelve
from typing import *
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict

# MUST BE BECORE IMPORTING TORCH
VISIBLE_DEVICES = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([x.split(':')[-1] for x in VISIBLE_DEVICES])
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import torch
import wandb
import prody as pr
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as scipy_rot

from utils.flow_matching_loss import FlowMatchingLossConfig, FlowMatchingLoss
from utils.dataset import UnclusteredProteinChainDataset, ClusteredDatasetSampler, idealize_backbone_coords, BatchData, ModelOutput
from utils.model import FlowModel, FlowModelModuleConfig
from utils.interpolant import Interpolant, InterpolantConfig
from utils.distributed_training import DistributedSamplerWrapper, setup_distributed
import utils.openfold_rigid_utils as ur 
import utils.utility_functions as uf

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


FLOW_MODEL_CONFIG = FlowModelModuleConfig(
    single_repr_node_embedding_dim = 256,
    pair_repr_node_embedding_dim = 128,
    node_positional_embedding_dim = 128,
    node_timestep_embedding_dim = 128,
    edge_repr_embedding_dim = 64,
    edge_num_distrogram_bins = 22,
    edge_embed_diffuse_mask = True,
    ipa_hidden_dim = 256,
    ipa_no_heads = 8,
    ipa_no_qk_points = 8,
    ipa_no_v_points = 12,
    num_trunk_blocks = 6,
    trunk_transformer_atten_heads = 4,
    trunk_transformer_num_layers = 2,
    dropout=0.1
)


INTERPOLANT_CONFIG = InterpolantConfig(
    min_t = 1e-2,
    twisting = False,
    rots_corrupt = True,
    rots_exp_rate = 10,
    trans_corrupt = True,
    trans_batch_ot = True,
    trans_sample_temp = 1.0,
    trans_vpsde_bmin = 0.1,
    trans_vpsde_bmax = 20.0,
)


FLOW_MATCHING_LOSS_CONFIG = FlowMatchingLossConfig(
    t_normalize_clip = 0.9,
    bb_atom_scale = 0.1,
    trans_scale = 0.1,
    translation_loss_weight = 2.0,
    rotation_loss_weights = 1.0,
    aux_loss_weight = 0.0,
    aux_loss_use_bb_loss = True,
    aux_loss_use_pair_loss = True,
    aux_loss_t_pass = 0.5,
)


def sampled_gly_coords_to_prody_prot(coords):
    """
    Takes a tensor of N, CA, C, O coordinates and returns a prody protein object.
    """
    # Determine the number of residues in the protein.
    num_res = coords.shape[0]

    # Flatten the coordinates.
    coords = coords.reshape(num_res * 4, -1).cpu().numpy()

    # Create metadata for each atom.
    names = ['N', 'CA', 'C', 'O'] * num_res
    atom_elements = [x[0] for x in names]
    resnums = [item for sublist in [[x] * 4 for x in range(num_res)] for item in sublist]
    resnames = ['GLY'] * (num_res * 4)
    chains = ['A'] * (num_res * 4)
    occupancies = [1.0] * (num_res * 4)

    # Create the prody protein object.
    output_protein = pr.AtomGroup('laser-flow Generated Structure')
    output_protein.setCoords(coords)
    output_protein.setNames(names)
    output_protein.setResnames(resnames)
    output_protein.setResnums(resnums)
    output_protein.setChids(chains)
    output_protein.setOccupancies(occupancies)
    output_protein.setElements(atom_elements)
    output_protein.setBetas([0.0] * len(atom_elements))

    return output_protein


def collate_fn(data_list: List[dict]) -> BatchData:
    max_length = 0
    all_batch_data = []
    for batch_idx, (complex_data, chain_key) in enumerate(data_list):
        complex_len = 0
        all_backbone_coords = []
        all_residue_frames = []
        curr_pdb_code = complex_data['pdb_code']
        for curr_chain_tup, chain_data in complex_data.items():

            if not isinstance(curr_chain_tup, tuple):
                continue

            is_sampled_chain = tuple(chain_key.split('-')[-2:]) == curr_chain_tup

            ideal_bb_coords = idealize_backbone_coords(chain_data['backbone_coords'].float(), chain_data['phi_psi_angles'])
            ca_com = uf.compute_center_of_mass(ideal_bb_coords[:, 1])
            centered_ideal_bb_coords = uf.center_coords(ideal_bb_coords, ca_com)
            centered_ideal_bb_coords = centered_ideal_bb_coords @ torch.from_numpy(scipy_rot.random().as_matrix().T).float()
            frames = uf.compute_residue_frames(centered_ideal_bb_coords)

            complex_len += centered_ideal_bb_coords.shape[0]
            all_residue_frames.append(frames)
            all_backbone_coords.append(centered_ideal_bb_coords)

        max_length = max(max_length, complex_len)
        complex_coords = torch.cat(all_backbone_coords, dim=0)
        complex_frames = torch.cat(all_residue_frames, dim=0)
        pre_padding_mask = torch.ones(complex_coords.shape[0])

        all_batch_data.append((complex_coords, complex_frames, pre_padding_mask))

    # Pad the first dimension of the data to the max length.
    padded_data = []
    for (coords, frames, mask) in all_batch_data:
        padding_len = max(max_length - coords.shape[0], 0)
        padded_data.append(
            (
                F.pad(coords, (0, 0, 0, 0, 0, padding_len), value=0.0),
                F.pad(frames, (0, 0, 0, 0, 0, padding_len), value=0.0),
                F.pad(mask, (0, padding_len), value=0.0)
            )
        )

    # Stack the batch data into a single tensor.
    padded_coords = torch.stack([x[0] for x in padded_data], dim=0)
    padded_frames = torch.stack([x[1] for x in padded_data], dim=0)
    padded_masks = torch.stack([x[2] for x in padded_data], dim=0)

    return BatchData(
        r_1 = ur.Rigid(ur.Rotation(rot_mats=padded_frames), padded_coords[:, :, 1]),
        res_mask = padded_masks,
        diffuse_mask = padded_masks,
        batch_size = padded_coords.shape[0],
        num_res = padded_coords.shape[1],
        res_indices = torch.arange(padded_coords.shape[1]).unsqueeze(0).expand(padded_coords.shape[0], -1),
        device=torch.device('cpu')
    )
    

def prepare_dataloader(rank, world_size, kwargs_dict, epoch_num, dataset, collate_fn, global_seed=None):
    """
    Initializes a distributed dataloader 
    """
    seed = epoch_num
    if global_seed is not None:
        seed += global_seed

    train_sampler = ClusteredDatasetSampler(dataset, is_test_dataset_sampler=False, seed=seed, **kwargs_dict)
    train_dist_sampler = DistributedSamplerWrapper(train_sampler, num_replicas=world_size, rank=rank, shuffle=False)
    train_dataloader = DataLoader(dataset, batch_sampler=train_dist_sampler, collate_fn=collate_fn) 

    # Sanity check that sampler returns same indices for all workers.
    # print(f'GPU-{rank}', list(train_sampler))
    return train_dataloader


class DistributedModelTrainer():
    def __init__(
        self, device_rank, world_size, flow_model_config, interpolant_config, flow_matching_loss_config, 
        dataset_shelve_path, metadata_shelve_path, device, 
        learning_rate = 1e-4, use_self_conditioning=True, use_wandb=True, debug=False,
        batch_size=500, max_protein_size=500
    ):
        self.debug = debug
        self.device_rank = device_rank
        self.world_size = world_size

        self.batch_size = batch_size
        self.max_protein_size = max_protein_size

        self.flow_model = FlowModel(flow_model_config)
        self.interpolant_config = interpolant_config
        self.flow_matching_loss_config = flow_matching_loss_config
        self.interpolant = Interpolant(interpolant_config)
        self.flow_matching_loss = FlowMatchingLoss(flow_matching_loss_config)
        self.optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=learning_rate)
        self.dataset = UnclusteredProteinChainDataset(dataset_shelve_path, metadata_shelve_path)

        self.device = device
        self.interpolant.set_device(device)
        self.flow_model = self.flow_model.to(device)
        self.flow_model = DistributedDataParallel(self.flow_model, device_ids=[device_rank], output_device=device_rank)

        self.learning_rate = learning_rate
        self.use_self_conditioning = use_self_conditioning

        self.epoch = 0
        self.training_epoch_metadata = defaultdict(list)
        self.validation_epoch_metadata = defaultdict(list)

        self.train_dataloader = self.get_dataloader()

        self.use_wandb = use_wandb
        if use_wandb and device_rank == 0:
            wandb_config = {}
            wandb_config.update(asdict(flow_model_config))
            wandb_config.update(asdict(interpolant_config))
            wandb_config.update(asdict(flow_matching_loss_config))
            wandb_config.update({'learning_rate': learning_rate, 'use_self_conditioning': use_self_conditioning, 'visible_devices': VISIBLE_DEVICES})

            wandb.init(project='laser-flow', entity='benf549', config=wandb_config)
    
    def get_dataloader(self):
        data_loader_kwargs = {
            'batch_size': self.batch_size, 'sample_randomly': True, 'max_protein_size': self.max_protein_size,
            'clustering_dataframe_path': '/nfs/polizzi/bfry/laser_clusters_remove_bromo.pkl',
            'subcluster_pickle_path': '/nfs/polizzi/bfry/laser_paper_analyses/pytorch_ligandmpnn_sampler/clusters_to_subcluster_data.pkl',
            'debug': False, 'subset_pdb_code_list': None
        }
        return prepare_dataloader(self.device_rank, self.world_size, data_loader_kwargs, self.epoch, self.dataset, collate_fn)

    def checkpoint(self):
        if not self.device_rank == 0:
            return

        checkpoint_dict = {
            'epoch': self.epoch,
            'flow_model_state_dict': self.flow_model.module.state_dict(),
            'interpolant_config': asdict(self.interpolant_config),
            'flow_matching_loss_config': asdict(self.flow_matching_loss_config),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'use_self_conditioning': self.use_self_conditioning
        }
        torch.save(checkpoint_dict, f'./model-checkpoints/checkpoint_epoch_{self.epoch:04d}.pt')
    
    def log(self):
         # Gather the logging data from all processes.
        all_logging_data = [None for _ in range(self.world_size)]
        dist.all_gather_object(all_logging_data, dict(self.training_epoch_metadata))

        # Rest the logging dict and exit if not the master process.
        self.training_epoch_metadata = defaultdict(list)
        if not self.device_rank == 0:
            return
        
        # Join the logging data from all processes.
        all_train_logging_data = defaultdict(list)
        for logging_data in all_logging_data:
            for key, value in logging_data.items():
                all_train_logging_data[key].extend(value)

        output = {'epoch': self.epoch}
        output.update({x: sum(y) / max(1, len(y)) for x,y in dict(all_train_logging_data).items()})
        print(output)
        if self.use_wandb:
            wandb.log(output)

    def train_epoch(self):
        self.flow_model.train()

        debug_break_flag = True
        for idx, batch in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader), dynamic_ncols=True, desc=f'Training Epoch {self.epoch}', disable=(self.device_rank != 0))):
            if self.debug and not debug_break_flag:
                break
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            corrupt_batch = self.interpolant.corrupt_batch(batch)

            # Implements self-conditioning by predicting the transformation once 
            #   and using it as input to the edge features in the prediction penalized in the loss.
            self_conditioning_trans = None
            if self.use_self_conditioning and (torch.rand((1,)).item() > 0.5):
                with torch.no_grad():
                    selfcond_output = self.flow_model(corrupt_batch)
                    self_conditioning_trans = (
                        selfcond_output.pred_trans * corrupt_batch.diffuse_mask[..., None] + corrupt_batch.trans_1 * (1 - corrupt_batch.diffuse_mask[..., None])
                    )
                    corrupt_batch._set_self_condition(self_conditioning_trans)

            # Predict maybe with self-conditioning, compute the loss, and step.
            model_out = self.flow_model(corrupt_batch)
            batch_losses = self.flow_matching_loss.compute_flow_matching_loss(corrupt_batch, model_out, reduce='mean')
            batch_losses.se3_vf_loss.backward()
            self.optimizer.step()

            # Record the losses for the current batch.
            for loss_component, loss in batch_losses.__dict__.items():
                self.training_epoch_metadata[loss_component].append(loss.item())

            if idx == 4:
                debug_break_flag = False
            
        self.epoch += 1
        self.train_dataloader = self.get_dataloader()

    @torch.no_grad()
    def sample_model(self, n_samples, sample_length, sample_timesteps):
        self.flow_model.eval()

        # Only sample from the master process.
        if not self.device_rank == 0:
            return

        try:
            # clean_sample_traj is stored as ['N', 'CA', 'C', 'CB', 'O']
            sample_trajectory, clean_sample_trajectory, clean_traj = self.interpolant.sample(self.flow_model, n_samples, sample_length, sample_timesteps)
        except Exception as e:
            print(f'Error in sampling: {e}')
            torch.cuda.empty_cache()
            return

        for sample_idx, sample_traj in enumerate([*clean_sample_trajectory[-1].unbind(dim=0)]):
            # Extract glycine backbone coordinates from the sample trajectory.
            gly_coords = sample_traj[:, [0, 1, 2, 4]] # N, CA, C, O

            # Write the sampled protein pdb file info to a stringio stream.
            output_str = io.StringIO()
            sampled_protein = sampled_gly_coords_to_prody_prot(gly_coords)
            pr.writePDBStream(output_str, sampled_protein)

            if self.use_wandb:
                wandb.log({ 'epoch': self.epoch, f'sample_{sample_idx}': wandb.Molecule(output_str, file_type='pdb') })

            # Write the sampled protein pdb file to disk.
            output_path = Path(f'.').resolve() / 'sampling' / f'sample_epoch_{self.epoch:04d}_sample_{sample_idx:02d}.pdb'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(output_str.getvalue())


def main(rank, world_size, params):
    # Initialize process group.
    setup_distributed(rank, world_size, params['master_port'])

    trainer = DistributedModelTrainer(
        rank, world_size, FLOW_MODEL_CONFIG, INTERPOLANT_CONFIG, FLOW_MATCHING_LOSS_CONFIG, 
        params['dataset_shelve_path'], params['metadata_shelve_path'], f'cuda:{rank}'
    )

    for _ in range(params['num_epochs']):
        trainer.train_epoch()
        trainer.log()
        if (trainer.epoch % 10 == 0):
            if not trainer.debug:
                trainer.checkpoint()
            trainer.sample_model(params['num_samples'], params['sample_length'], params['sample_timesteps'])


if __name__ == "__main__":
    file_path = Path('..')
    print((str(file_path.resolve() / 'laser_training_database' / 'all_data_shelf_hbond_sconly_rigorous.db')))
    params = {
        'dataset_shelve_path': file_path / 'laser_training_database' / 'all_data_shelf_hbond_sconly_rigorous.db',
        'metadata_shelve_path': file_path / 'laser_training_database' / 'pdb_metadata_shelf_addhaslig_perchain.db',
        'num_epochs': 500,
        'num_samples': 1,
        'sample_length': 128,
        'sample_timesteps': 100,
        'master_port': '56889'
    }
    world_size = len(VISIBLE_DEVICES)
    mp.spawn(main, args=(world_size, params), nprocs=world_size)
