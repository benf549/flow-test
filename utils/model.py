
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import truncnorm
from dataclasses import dataclass, asdict, field

from utils.constants import NM_TO_ANG_SCALE, ANG_TO_NM_SCALE
from utils.openfold_rigid_utils import Rigid
from utils.dataset import BatchData, ModelOutput


@dataclass
class FlowModelModuleConfig:
    """
    Main configuration object for the flow model.
    """
    single_repr_node_embedding_dim: int
    pair_repr_node_embedding_dim: int

    node_positional_embedding_dim: int
    node_timestep_embedding_dim: int

    edge_repr_embedding_dim: int
    edge_num_distrogram_bins: int
    edge_embed_diffuse_mask: bool

    ipa_hidden_dim: int
    ipa_no_heads: int
    ipa_no_qk_points: int
    ipa_no_v_points: int

    num_trunk_blocks: int
    trunk_transformer_atten_heads: int
    trunk_transformer_num_layers: int

    dropout: float

    node_embedding_config: 'NodeEmbeddingConfig' = field(init=False)
    edge_embedding_config: 'EdgeEmbeddingConfig' = field(init=False)
    trunk_config_generator: Callable[[int], 'FlowModelTrunkConfig'] = field(init=False)

    def __post_init__(self):
        self.node_embedding_config = NodeEmbeddingConfig(
            c_s = self.single_repr_node_embedding_dim, c_pos_emb = self.node_positional_embedding_dim,
            c_timestep_emb = self.node_timestep_embedding_dim
        )

        self.edge_embedding_config = EdgeEmbeddingConfig(
            c_s = self.single_repr_node_embedding_dim, c_p = self.pair_repr_node_embedding_dim,
            feat_dim = self.edge_repr_embedding_dim, num_bins = self.edge_num_distrogram_bins,
            embed_diffuse_mask = self.edge_embed_diffuse_mask
        )

        ipa_config = IPA_Config(
            c_s = self.single_repr_node_embedding_dim, c_z = self.pair_repr_node_embedding_dim,
            c_hidden = self.ipa_hidden_dim, no_heads = self.ipa_no_heads,
            no_qk_points= self.ipa_no_qk_points, no_v_points = self.ipa_no_v_points
        )

        # Create a function that allows us to set final_layer flag for the last trunk block.
        self.trunk_config_generator = lambda layer_idx: FlowModelTrunkConfig(
            single_repr_transformer_atten_heads=self.trunk_transformer_atten_heads,
            single_repr_transformer_num_layers=self.trunk_transformer_num_layers,
            dropout=self.dropout, ipa_config=ipa_config,
            final_layer=(layer_idx == (self.num_trunk_blocks - 1)), 
        ) 


@dataclass
class NodeEmbeddingConfig:
    c_s: int
    c_pos_emb: int
    c_timestep_emb: int
    embed_size: int = field(init=False)

    def __post_init__(self):
        self.embed_size = self.c_pos_emb + self.c_timestep_emb * 2 + 1


@dataclass
class EdgeEmbeddingConfig:
    c_s: int
    c_p: int
    feat_dim: int
    num_bins: int
    embed_diffuse_mask: bool
    total_edge_feats: int = field(init=False)

    def __post_init__(self):
        self.total_edge_feats = self.feat_dim * 3 + self.num_bins * 2 
        self.total_edge_feats += 2 if self.embed_diffuse_mask else 0


@dataclass
class IPA_Config:
    c_s: int
    c_z: int
    c_hidden: int
    no_heads: int
    no_qk_points: int
    no_v_points: int
    inf: float = 1e5
    eps: float = 1e-5


@dataclass
class FlowModelTrunkConfig:
    single_repr_transformer_atten_heads: int
    single_repr_transformer_num_layers: int
    dropout: float
    final_layer: bool
    ipa_config: IPA_Config


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = np.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * np.pi / (max_len**(2*K[None]/embed_size))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * np.pi / (max_len**(2*K[None]/embed_size))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1) # type: ignore
    return pos_embedding


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = np.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace( min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


class CustomLinear(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, init: str = 'default'):
        super(CustomLinear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)
        
        if init == 'default':
            lecun_normal_init_(self.weight)
        elif init == 'relu':
            he_normal_init_(self.weight)
        elif init == 'final':
            final_init_(self.weight)
        else:
            raise NotImplementedError(f"Initialization method {init} not implemented.")


class NodeFeatureNet(nn.Module):
    """
    Embeds the nodes with positional and time-step encodings.
    """
    def __init__(self, module_cfg: NodeEmbeddingConfig):
        super(NodeFeatureNet, self).__init__()
        self.c_s = module_cfg.c_s
        self.c_pos_emb = module_cfg.c_pos_emb
        self.c_timestep_emb = module_cfg.c_timestep_emb
        self.embed_size = module_cfg.embed_size
        self._cfg = module_cfg

        self.linear = nn.Linear(self.embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, so3_t, r3_t, res_mask, diffuse_mask, pos):
        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            self.embed_t(so3_t, res_mask),
            self.embed_t(r3_t, res_mask)
        ]
        return self.linear(torch.cat(input_feats, dim=-1))


class EdgeFeatureNet(nn.Module):
    def __init__(self, module_cfg: EdgeEmbeddingConfig):
        super(EdgeFeatureNet, self).__init__()

        self.c_s = module_cfg.c_s
        self.c_p = module_cfg.c_p
        self.feat_dim = module_cfg.feat_dim
        self.total_edge_feats = module_cfg.total_edge_feats
        self._cfg = module_cfg

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        self.edge_embedder = nn.Sequential(
            nn.Linear(self.total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, r):
        d = r[:, :, None] - r[:, None, :]
        pos_emb = get_index_embedding(d, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, s, t, sc_t, p_mask, diffuse_mask):
        # Input: [b, n_res, c_s]
        num_batch, num_res, _ = s.shape

        # [b, n_res, c_p]
        p_i = self.linear_s_p(s)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)

        # [b, n_res]
        r = torch.arange(num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        relpos_feats = self.embed_relpos(r)

        dist_feats = calc_distogram(t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        # TODO: Handle self-conditioning features.
        sc_feats = calc_distogram(sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)

        all_edge_feats = [cross_node_feats, relpos_feats, dist_feats, sc_feats]
        if self._cfg.embed_diffuse_mask:
            diff_feat = self._cross_concat(diffuse_mask[..., None], num_batch, num_res)
            all_edge_feats.append(diff_feat)
        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats


class EdgeTransition(nn.Module):
    def __init__( 
        self, *, node_embed_size, edge_embed_in, edge_embed_out, 
        num_layers=2, node_dilation=2
    ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = CustomLinear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(CustomLinear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = CustomLinear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat([edge_embed, edge_bias], axis=-1).reshape(batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed


class IPA(nn.Module):
    def __init__(self, ipa_config: IPA_Config):
        super(IPA, self).__init__()
        self.c_s = ipa_config.c_s
        self.c_z = ipa_config.c_z
        self.c_hidden = ipa_config.c_hidden
        self.no_heads = ipa_config.no_heads
        self.no_qk_points = ipa_config.no_qk_points
        self.no_v_points = ipa_config.no_v_points
        self.inf = ipa_config.inf
        self.eps = ipa_config.eps

        hc = self.c_hidden * self.no_heads
        self.linear_q = CustomLinear(self.c_s, hc)
        self.linear_kv = CustomLinear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = CustomLinear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = CustomLinear(self.c_s, hpkv)

        self.linear_b = CustomLinear(self.c_z, self.no_heads)
        self.down_z = CustomLinear(self.c_z, self.c_z // 4)

        concat_out_dim = (self.c_z // 4 + self.c_hidden + self.no_v_points * 4)

        # Initialize to all zeros.
        self.linear_out = CustomLinear(self.no_heads * concat_out_dim, self.c_s, init='final')

        self.head_weights = nn.Parameter(torch.zeros((self.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(self, s, z, r, mask):
        #######################################
        # Generate scalar and point activations
        #######################################
        # Compute scalar qkvs.
        q = self.linear_q(s) # [*, N_res, H * C_hidden]
        kv = self.linear_kv(s)

        # Expand tensors to number of heads and break up kv into k and v.
        q = q.view(q.shape[:-1] + (self.no_heads, -1)) # [*, N_res, H, C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1)) # [*, N_res, H, 2 * C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1) # [*, N_res, H, C_hidden]

        # Compute point attention qkvs.
        q_pts = self.linear_q_points(s) # Converts scalar activations to a learned set of query 3D points.
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1) # Stacks x,y,z into the last dimension.
        # Convert learnable points in the global frame to the local frames.
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        ) # Separates the 3D query points into the number of heads, points, and coords.

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s) # Converts scalar activations to a learned set of key/value 3D points.

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1) # Stacks x,y,z into the last dimension.
        # Rotate the key and value points to the local frame?
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(
            kv_pts.shape[:-2] + (self.no_heads, -1, 3)
        ) # Separates the 3D key/value points into the number of heads, points, and coords.

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        ) # Separates the key and value points into the number of query and value points.

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        # Transforms the pairwise features.
        b = self.linear_b(z[0])

        # Q^T@K for each head.
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        # Scale the scalars proportional to the number of hidden units.
        a *= np.sqrt(1.0 / (3 * self.c_hidden))

        # Skip connection with pairwise features.
        a += (np.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        # Compute displacements between query and key points.
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)

        # [*, N_res, N_res, H, P_q]
        pt_att = (pt_displacement ** 2).sum(dim=-1)

        # Learnable weight for contribution of information for each head from pt_att to the attention score.
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )

        # Scale point-attention coefficients proportional to the number of query points.
        head_weights = head_weights * np.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        # Compute a mask to prevent attending to hidden residues
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res], swaps heads to the front
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        # Compute masked attention.
        a = a + pt_att 
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        # [atten = softmax(QK^T)] V
        o = torch.matmul(
            a, v.transpose(-2, -3)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        # Concates heads with embedding channels.
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v] 
        # Attends to the point values.
        o_pt = torch.sum(
            ( a[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ), dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        # Converts scaled point vectors back from local to global frame
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        # Compute length of final point vectors as additional feature for each head.
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        # Stack all the heads' points together.
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, N_res, H, C_z // 4]
        # Compute attention-gated information from the input pair-representation by down-projecting.
        pair_z = self.down_z(z)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        # Stack all heads' down-projected info together.
        o_pair = flatten_final_dims(o_pair, 2)

        # Gather together all the info to pass through output linear layer.
        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        # Pass all the information generated through a linear layer which learns to weight the contributions.
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            )
        )
        return s


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransition, self).__init__()
        self.c = c

        self.linear_1 = CustomLinear(self.c, self.c, init="relu")
        self.linear_2 = CustomLinear(self.c, self.c, init="relu")
        self.linear_3 = CustomLinear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)
        return s


class BackboneUpdate(nn.Module):
    def __init__(self, c_s: int):
        super(BackboneUpdate, self).__init__()
        self.c_s = c_s
        self.linear = CustomLinear(self.c_s, 6, init="final")
    
    def forward(self, s):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector quaterion (b, c, d) coeffs and (x, y, z) translation.
                (see af2 supplement.)
        """
        update = self.linear(s)
        return update


class FlowModelTrunk(nn.Module):
    def __init__(self, flow_model_trunk_config: FlowModelTrunkConfig):
        super(FlowModelTrunk, self).__init__()
        self.ipa_config = flow_model_trunk_config.ipa_config
        self.is_final_layer = flow_model_trunk_config.final_layer

        self.tfmr_num_layers = flow_model_trunk_config.single_repr_transformer_num_layers
        self.tfmr_num_atten_heads = flow_model_trunk_config.single_repr_transformer_atten_heads

        self.ipa = IPA(self.ipa_config)
        self.ipa_ln = nn.LayerNorm(self.ipa_config.c_s)
        self.smt = StructureModuleTransition(self.ipa_config.c_s)

        trans_enc = nn.TransformerEncoderLayer(
            nhead = self.tfmr_num_atten_heads, 
            d_model = self.ipa_config.c_s,
            dim_feedforward = self.ipa_config.c_s,
            batch_first=True, dropout=0.0, norm_first=False,
        )

        self.seq_tfmr = nn.TransformerEncoder(trans_enc, self.tfmr_num_layers, enable_nested_tensor=False)
        self.post_tfmr = CustomLinear(self.ipa_config.c_s, self.ipa_config.c_s, init="final")

        self.backbone_update = BackboneUpdate(self.ipa_config.c_s)
        if not self.is_final_layer:
            self.edge_transition = EdgeTransition(
                node_embed_size=self.ipa_config.c_s, 
                edge_embed_in=self.ipa_config.c_z, 
                edge_embed_out=self.ipa_config.c_z,
            )

        dropout = flow_model_trunk_config.dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, s: torch.Tensor, z: torch.Tensor, r: Rigid, node_mask, diffuse_mask, edge_mask):
        node_embeddings, pair_embeddings = s, z

        ipa_embed = self.ipa(node_embeddings, pair_embeddings, r, node_mask)
        ipa_embed = ipa_embed * node_mask[..., None]
        node_embeddings = self.ipa_ln(self.dropout(node_embeddings + ipa_embed))
        seq_tfmr_out = self.seq_tfmr(node_embeddings, src_key_padding_mask=(1 - node_mask).bool())
        node_embeddings = node_embeddings + self.post_tfmr(seq_tfmr_out)
        node_embeddings = self.smt(node_embeddings) * node_mask[..., None]

        rigid_update = self.backbone_update(node_embeddings)
        r = r.compose_q_update_vec(rigid_update, (node_mask * diffuse_mask)[..., None])

        if hasattr(self, 'edge_transition'):
            pair_embeddings = self.edge_transition(node_embeddings, pair_embeddings)
            pair_embeddings = pair_embeddings * edge_mask[..., None]

        return node_embeddings, pair_embeddings, r


class FlowModel(nn.Module):
    def __init__(self, flow_model_config: FlowModelModuleConfig):
        super(FlowModel, self).__init__()
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * NM_TO_ANG_SCALE) 

        self.node_embedding_layer = NodeFeatureNet(flow_model_config.node_embedding_config)
        self.edge_embedding_layer = EdgeFeatureNet(flow_model_config.edge_embedding_config)
        self.trunk = nn.ModuleList([
            FlowModelTrunk(flow_model_config.trunk_config_generator(trunk_idx)) 
            for trunk_idx in range(flow_model_config.num_trunk_blocks)
        ])
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch_data: BatchData) -> ModelOutput:
        # Get translation from rigid.
        r_trans = batch_data.r_t.get_trans()

        # Uses self-conditioning information in the edge update if available.
        self_conditioning_trans = batch_data.trans_self_conditioning
        if self_conditioning_trans is None:
            self_conditioning_trans = torch.zeros_like(r_trans) 
        edge_mask = batch_data.res_mask[:, None] * batch_data.res_mask[:, :, None]

        # TODO: figure out what diffuse_mask is.
        s = self.node_embedding_layer(
            batch_data.so3_t, batch_data.r3_t, batch_data.res_mask, batch_data.diffuse_mask, batch_data.res_indices
        )
        z = self.edge_embedding_layer(s, r_trans, self_conditioning_trans, edge_mask, batch_data.diffuse_mask)

        # Nanometer scale distances are better for IPA according to AF2 paper.
        r = self.rigids_ang_to_nm(batch_data.r_t)
        for module in self.trunk:
            s, z, r = module(s, z, r, batch_data.res_mask, batch_data.diffuse_mask, edge_mask)
        r = self.rigids_nm_to_ang(r)

        return ModelOutput(pred_trans=r.get_trans(), pred_rotmats=r.get_rots().get_rot_mats())
