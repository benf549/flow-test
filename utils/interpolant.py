from typing import *

import sys
import torch
from tqdm import tqdm 
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

from torch import autograd
import scipy.spatial.transform as scipy_transform
from dataclasses import dataclass

from utils.openfold_rigid_utils import Rigid, Rotation
from utils.constants import NM_TO_ANG_SCALE
from utils.dataset import BatchData
from utils.model import FlowModel
import utils.so3_utils as so3_utils
from utils.all_atom import transrot_to_atom37

# from motif_scaffolding import twisting

@dataclass
class InterpolantConfig:
    min_t: float
    twisting: bool

    rots_corrupt: bool
    rots_exp_rate: float

    trans_corrupt: bool
    trans_batch_ot: bool
    trans_sample_temp: float
    trans_vpsde_bmin: float
    trans_vpsde_bmax: float


class Interpolant:
    def __init__(self, cfg: InterpolantConfig):
        self.min_t = cfg.min_t
        self.twisting = cfg.twisting
        self.trans_corrupt = cfg.trans_corrupt
        self.trans_sample_temp = cfg.trans_sample_temp
        self.trans_vpsde_bmin = cfg.trans_vpsde_bmin
        self.trans_vpsde_bmax = cfg.trans_vpsde_bmax
        self.rots_corrupt = cfg.rots_corrupt
        self.rots_exp_rate = cfg.rots_exp_rate
        self.trans_batch_ot = cfg.trans_batch_ot

        self._igso3 = None

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir='.cache')
        return self._igso3
    
    def set_device(self, device):
        self._device = device
    
    def corrupt_batch(self, batch: BatchData) -> BatchData:
        output_batch = deepcopy(batch)

        # Sample a value between 0 and 1 for each batch element.
        t = self.sample_t(batch.batch_size)[:, None] # [B, 1]
        so3_t, r3_t = t, t

        # Apply corruptions to each manifold independently.
        trans_t = output_batch.trans_1
        if self.trans_corrupt:
            # Compute frame location at time t.
            trans_t = self._corrupt_trans(
                output_batch.trans_1, r3_t, output_batch.res_mask, output_batch.diffuse_mask
            )
        rot_t = output_batch.rotmats_1
        if self.rots_corrupt:
            # Compute rotation matrices at time t.
            rotmats_t = self._corrupt_rotmats(
                output_batch.rotmats_1, so3_t, output_batch.res_mask, output_batch.diffuse_mask
            )
            assert not torch.any(torch.isnan(rotmats_t)), "NaNs in rotmats_t."

        output_rigid = Rigid(Rotation(rot_mats=rot_t), trans_t)
        output_batch._update_from_corrupt(output_rigid, so3_t, r3_t)

        return output_batch
    
    def sample(self, model: FlowModel, num_batch: int, num_res: int, num_sample_timesteps: int, self_condition: bool = True):
        if not hasattr(self, "_device"):
            raise ValueError("Device not set, call interpolant.set_device(device) first.")

        # Set-up initial prior samples
        trans_0 = _centered_gaussian(num_batch, num_res, self._device) * NM_TO_ANG_SCALE
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        prot_traj = [(trans_0, rotmats_0)] # Store flow trajectory.

        # Set-up time
        clean_traj = []
        ts = torch.linspace(self.min_t, 1.0, num_sample_timesteps)
        t_1 = ts[0]
        ### Integrate forward from t_0 to min_t
        for t_2 in tqdm(ts[1:], total=num_sample_timesteps-1):
            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            sample_batch = self.get_sample_batch_time_t(num_batch, num_res, trans_t_1, rotmats_t_1, t_1)
            with torch.no_grad():
                sampling_out = model(sample_batch)
                
            if self_condition:
                sample_batch._set_self_condition(sampling_out.pred_trans)
            
            # Process model output.
            clean_traj.append((sampling_out.pred_trans.detach().cpu(), sampling_out.pred_rotmats.detach().cpu()))

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(d_t, t_1, sampling_out.pred_trans, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(d_t, t_1, sampling_out.pred_rotmats, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        sample_batch = self.get_sample_batch_time_t(num_batch, num_res, trans_t_1, rotmats_t_1, t_1)
        with torch.no_grad():
            final_out = model(sample_batch)
        clean_traj.append(
            (final_out.pred_trans.detach().cpu(), final_out.pred_rotmats.detach().cpu())
        )
        prot_traj.append((final_out.pred_trans, final_out.pred_rotmats))

        # Convert trajectories to atom37.
        atom37_traj = transrot_to_atom37(prot_traj, sample_batch.res_mask)
        clean_atom37_traj = transrot_to_atom37(clean_traj, sample_batch.res_mask)

        return atom37_traj, clean_atom37_traj, clean_traj

    
    def get_sample_batch_time_t(self, num_batch, num_res, trans_t, rotmats_t, t):
        batch = BatchData(
            batch_size=num_batch,
            num_res=num_res,
            res_mask = torch.ones((num_batch, num_res), device=self._device),
            r_1=None,
            diffuse_mask=torch.ones((num_batch, num_res), device=self._device),
            device=self._device,
            res_indices=torch.arange(num_res, device=self._device),
        )
        batch._update_from_corrupt(
            r_t=Rigid(Rotation(rot_mats=rotmats_t), trans=trans_t),
            so3_t=torch.full((num_batch, 1), t, device=self._device),
            r3_t=torch.full((num_batch, 1), t, device=self._device),
        )
        return batch

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        # Take linear step towards t=1.
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return trans_t + trans_vf * d_t
    
    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        # Take exponential step towards t=1.
        scaling = self.rots_exp_rate
        return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)
    

    def sample_t(self, num_batch):
        if not hasattr(self, "_device"):
            raise ValueError("Device not set, call interpolant.set_device(device) first.")

        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self.min_t) + self.min_t
    
    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
        if not hasattr(self, "_device"):
            raise ValueError("Device not set, call interpolant.set_device(device) first.")

        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device) # type: ignore
        trans_0 = trans_nm_0 * NM_TO_ANG_SCALE
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]
    
    def _corrupt_rotmats(self, r_1, t, res_mask, diffuse_mask):
        if not hasattr(self, "_device"):
            raise ValueError("Device not set, call interpolant.set_device(device) first.")
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", r_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], r_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, r_1, diffuse_mask)


def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True) # type: ignore


def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        scipy_transform.Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )