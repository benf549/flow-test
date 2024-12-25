import torch
import torch.nn.functional as F

def compute_center_of_mass(ca_coords):
    return torch.mean(ca_coords, dim=0)

def center_coords(coords, com):
    return coords - com


def compute_residue_frames(bb_coords):
    """
    Computes protein backbone frame rotation matrices.

    This function computes a local reference frame for each residue from the 
    backbone coordinates provided. The frames are constructed using the 
    Gram-Schmidt orthogonalization process and results in matrices with 
    orthonormal basis vectors for each residue's backbone frame.

    Args:
    - bb_coords: A tensor of shape (N, M) where N is the number of residues 
      and M is at least 4, representing the coordinates of backbone atoms.

    Returns:
    - frame: A tensor of shape (N, 3, 3) representing rotation matrices for 
      each residue. Each rotation matrix is constructed as follows:

      Given backbone coordinates that include the nitrogen (N), alpha carbon (Ca),
      and carbon (C) atoms:
      
      1. Compute vectors:
         - v1 = N - Ca
         - v2 = C - Ca

      2. Gram-Schmidt orthogonalization to produce orthonormal basis:
         - e1 = normalize(v1)
         - u2 = v2 - dot(e1, v2) * e1
         - e2 = normalize(u2)
         - e3 = cross(e1, e2)

      Each resulting (3x3) matrix for a residue is formed by stacking the 
      orthonormal vectors:
      
      [[ e1_x, e2_x, e3_x ],
       [ e1_y, e2_y, e3_y ],
       [ e1_z, e2_z, e3_z ]]
      
      This stack results in a column-major order tensor for each residue frame.
    """
    # Expand the frames to the size of the edges (E, 3).
    N, Ca, C = bb_coords[:, [0, 1, 3]].unbind(dim=1)

    # 'Gram-Schmidt' orthogonalization
    v1 = N - Ca 
    v2 = C - Ca 
    e1 = F.normalize(v1, dim=-1)  # (N, 3)
    e1_v2_dot = torch.einsum("ni, ni -> n", e1, v2)[:, None]  # (N, 1)
    u2 = v2 - e1 * e1_v2_dot  # (N, 3)
    e2 = torch.nn.functional.normalize(u2, dim=-1)  # (N, 3)
    e3 = torch.cross(e1, e2, dim=-1)  # (N, 3)
    frame = torch.stack((e1, e2, e3), dim=-1) # (N, 3, 3)
    
    return frame