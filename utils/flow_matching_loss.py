from typing import *
import torch
from dataclasses import dataclass

import utils.so3_utils as uso3
import utils.all_atom as uaa
from utils.dataset import BatchData, ModelOutput

@dataclass
class FlowMatchingLossComponents:
    trans_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    rots_vf_loss: torch.Tensor
    se3_vf_loss: torch.Tensor

    def mean_along_batch(self):
        self.trans_loss = torch.mean(self.trans_loss)
        self.auxiliary_loss = torch.mean(self.auxiliary_loss)
        self.rots_vf_loss = torch.mean(self.rots_vf_loss)
        self.se3_vf_loss = torch.mean(self.se3_vf_loss)


@dataclass
class FlowMatchingLossConfig:
    t_normalize_clip: float 
    bb_atom_scale: float
    trans_scale: float
    translation_loss_weight: float
    rotation_loss_weights: float
    aux_loss_weight: float
    aux_loss_use_bb_loss: bool
    aux_loss_use_pair_loss: bool
    aux_loss_t_pass: float


class FlowMatchingLoss:
    def __init__(self, flow_matching_loss: FlowMatchingLossConfig):
        self.t_normalize_clip = flow_matching_loss.t_normalize_clip
        self.bb_atom_scale = flow_matching_loss.bb_atom_scale
        self.trans_scale = flow_matching_loss.trans_scale
        self.translation_loss_weight = flow_matching_loss.translation_loss_weight
        self.rotation_loss_weights = flow_matching_loss.rotation_loss_weights
        self.aux_loss_weight = flow_matching_loss.aux_loss_weight
        self.aux_loss_use_bb_loss = flow_matching_loss.aux_loss_use_bb_loss
        self.aux_loss_use_pair_loss = flow_matching_loss.aux_loss_use_pair_loss
        self.aux_loss_t_pass = flow_matching_loss.aux_loss_t_pass

    def compute_flow_matching_loss(
            self, batch_data: BatchData, model_out: ModelOutput, reduce: Optional[str] = None
    ) -> FlowMatchingLossComponents:
        loss_mask = batch_data.res_mask * batch_data.diffuse_mask
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Found empty batch!')
        
        # [B, N, 3]
        gt_rot_vf = uso3.calc_rot_vf(batch_data.rotmats_t, batch_data.rotmats_1.type(torch.float32))
        if torch.any(torch.isnan(gt_rot_vf)):
            raise ValueError('NaNs in gt_rot_vf.')
        gt_bb_atoms = uaa.to_atom37(batch_data.trans_1, batch_data.rotmats_1)[:, :, :3] 
        
        # Timestep used for normalization.
        r3_norm_scale = 1 - torch.min(batch_data.r3_t[..., None], torch.tensor(self.t_normalize_clip))
        so3_norm_scale = 1 - torch.min(batch_data.so3_t[..., None], torch.tensor(self.t_normalize_clip))

        # Compute losses
        pred_rots_vf = uso3.calc_rot_vf(batch_data.rotmats_t, model_out.pred_rotmats)
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError('NaNs in pred_rots_vf.')

        # Backbone atom loss
        pred_bb_atoms = uaa.to_atom37(model_out.pred_trans, model_out.pred_rotmats)[:, :, :3]
        gt_bb_atoms *= self.bb_atom_scale / r3_norm_scale[..., None]
        pred_bb_atoms *= self.bb_atom_scale / r3_norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None], dim=(-1, -2, -3)
        ) / loss_denom

        # Translation VF loss
        trans_error = (batch_data.trans_1 - model_out.pred_trans) / r3_norm_scale * self.trans_scale
        trans_loss = self.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None], dim=(-1, -2)
        ) / loss_denom
        trans_loss = torch.clamp(trans_loss, max=5)

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
        rots_vf_loss = self.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None], dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([batch_data.batch_size, batch_data.num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_bb_atoms.reshape([batch_data.batch_size, batch_data.num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
            )

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([batch_data.batch_size, batch_data.num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([batch_data.batch_size, batch_data.num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) + 1)

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (
            bb_atom_loss * self.aux_loss_use_bb_loss
            + dist_mat_loss * self.aux_loss_use_pair_loss
        )
        auxiliary_loss *= (
            (batch_data.r3_t[:, 0] > self.aux_loss_t_pass)
            & (batch_data.so3_t[:, 0] > self.aux_loss_t_pass)
        )
        auxiliary_loss *= self.aux_loss_weight
        auxiliary_loss = torch.clamp(auxiliary_loss, max=5)

        se3_vf_loss += auxiliary_loss
        if torch.any(torch.isnan(se3_vf_loss)):
            raise ValueError('NaNs in se3_vf_loss.')

        output_loss_components = FlowMatchingLossComponents( 
            trans_loss=trans_loss, auxiliary_loss=auxiliary_loss,
            rots_vf_loss=rots_vf_loss, se3_vf_loss=se3_vf_loss,
        )
        if reduce is not None:
            if reduce == 'mean':
                output_loss_components.mean_along_batch()
            else:
                raise ValueError(f'Invalid reduce value: {reduce}')
        return output_loss_components
