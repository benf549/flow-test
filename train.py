import shelve
from pathlib import Path
from typing import *

import torch

from utils.flow_matching_loss import FlowMatchingLossConfig, FlowMatchingLoss
from utils.dataset import UnclusteredProteinChainDataset, idealize_backbone_coords, BatchData, ModelOutput
from utils.model import FlowModel, FlowModelModuleConfig
from utils.interpolant import Interpolant, InterpolantConfig
import utils.openfold_rigid_utils as ur 
import utils.utility_functions as uf
from dataclasses import dataclass


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

def get_dummy_data(dataset):
    able_chain_A, _ = dataset[dataset.chain_key_to_index['6w70_1-A-A']]
    bb_coords = able_chain_A[('A', 'A')]['backbone_coords']
    phi_psi_angles = able_chain_A[('A', 'A')]['phi_psi_angles']

    ideal_bb_coords = idealize_backbone_coords(bb_coords, phi_psi_angles)
    # TODO: center on ligand.
    ca_com = uf.compute_center_of_mass(ideal_bb_coords[:, 1])
    centered_ideal_bb_coords = uf.center_coords(ideal_bb_coords, ca_com)
    frames = uf.compute_residue_frames(centered_ideal_bb_coords)

    return frames, centered_ideal_bb_coords, ca_com


def get_dummy_batch(dataset) -> BatchData:
    target_frames, target_bb_coords, target_ca_com = get_dummy_data(dataset)
    target_frames_rigid = ur.Rigid(ur.Rotation(rot_mats=target_frames), target_bb_coords[:, 1])[None, ...]

    # Create dummy masks for now.
    batch_size, num_residues = target_frames_rigid.get_trans().shape[:2]
    node_mask = torch.ones(batch_size, num_residues, dtype=torch.float)
    diffuse_mask = node_mask.clone()

    res_indices = torch.arange(num_residues)[None, ...]

    return BatchData(
        r_1 = target_frames_rigid, res_mask = node_mask, diffuse_mask = diffuse_mask,
        batch_size = batch_size, num_res = num_residues, device = target_frames_rigid.device,
        res_indices = res_indices
    )

def train_epoch(flow_model, interpolant, flow_matching_loss, optimizer, dataset, use_self_conditioning):
    batch = get_dummy_batch(dataset)
    interpolant.set_device(batch.device)

    optimizer.zero_grad()
    corrupt_batch = interpolant.corrupt_batch(batch)

    # Implements self-conditioning by predicting the transformation once 
    #   and using it as input to the edge features in the prediction penalized in the loss.
    self_conditioning_trans = None
    if use_self_conditioning and (torch.rand((1,)).item() > 0.5):
        with torch.no_grad():
            selfcond_output = flow_model(corrupt_batch)
            self_conditioning_trans = (
                selfcond_output.pred_trans * corrupt_batch.diffuse_mask[..., None]
                + corrupt_batch.trans_1 * (1 - corrupt_batch.diffuse_mask[..., None])
            )
            corrupt_batch._set_self_condition(self_conditioning_trans)

    model_out = flow_model(corrupt_batch)

    batch_losses = flow_matching_loss.compute_flow_matching_loss(corrupt_batch, model_out, reduce='mean')
    batch_losses.se3_vf_loss.backward()
    optimizer.step()

    return batch_losses


def sample_model(flow_model, interpolant):
    sample_trajectory, clean_sample_trajectory = interpolant.sample(flow_model, 1, 100, 10)
    return sample_trajectory, clean_sample_trajectory



def main(params, use_self_conditioning = True, learning_rate = 1e-4):
    flow_model = FlowModel(FLOW_MODEL_CONFIG)
    interpolant = Interpolant(INTERPOLANT_CONFIG)
    flow_matching_loss = FlowMatchingLoss(FLOW_MATCHING_LOSS_CONFIG)
    dataset = UnclusteredProteinChainDataset(params['dataset_shelve_path'], params['metadata_shelve_path'])
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=learning_rate)

    # Get dummy data target frames.
    for epoch in range(1, 101):
        for _ in range(10):
            batch_losses = train_epoch(flow_model, interpolant, flow_matching_loss, optimizer, dataset, use_self_conditioning)

        # List of BxNx37x3 tensors containing N, CA, C, CB, and O coords (in that order)
        _, clean_sample_trajectory = sample_model(flow_model, interpolant)
        print(clean_sample_trajectory[0][:, :, :5])

        print(f'Epoch: {epoch * 10}')
        for k,v in vars(batch_losses).items():
            print(f'{k}: {v.item()}')
        print()

        raise NotImplementedError


if __name__ == "__main__":
    file_path = Path(__file__).parent
    params = {
        'dataset_shelve_path': file_path / 'dataset' / 'dataset_shelve',
        'metadata_shelve_path': file_path / 'dataset' / 'metadata_shelve',
    }
    main(params)