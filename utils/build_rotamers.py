"""
Implements a RotamerBuilder class that builds full atom models of proteins given chi angles.
Should be used with atom orderings as defined in constants.py.

Benjamin Fry (bfry@g.harvard.edu) 
12/6/2023
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, Union
from utils.constants import MAX_NUM_RESIDUE_ATOMS, MAX_NUM_PROTONATED_RESIDUE_ATOMS, MAX_NUM_TRIPLETS_PER_RESIDUE, CHI_BIN_MIN, CHI_BIN_MAX, COVALENT_HYDROGEN_BOND_MAX_DISTANCE, ideal_aa_coords, ideal_bond_lengths, ideal_bond_angles, aa_to_chi_angle_atom_index, aa_to_leftover_atoms, alignment_indices, aa_short_to_idx, ideal_prot_aa_coords, aa_to_hydrogen_alignment_triad_indices, aa_to_hydrogen_alignment_index, optional_hydrogen_map, hydrogen_extended_dataset_atom_order, aa_to_chi_angle_mask, atom_to_atomic_number

class RotamerBuilder(nn.Module):
    """
    A class for building full atom models of proteins given chi angles.
    Also used for embedding chi angles in an angular RBF.
    Built into a torch module to put all of the helper tensors on the GPU if necessary.
    """
    def __init__(self, chi_angle_rbf_bin_width, **kwargs):
        super(RotamerBuilder, self).__init__()

        # Set the width of the bins for the chi angles.
        self.chi_angle_rbf_bin_width = chi_angle_rbf_bin_width

        # Pad ideal coords with NaNs to allow for indexing with MAX_NUM_RESIDUE_ATOMS indices.
        gly_ideal_coords = ideal_aa_coords[aa_short_to_idx['G']].unsqueeze(0)
        gly_ideal_bls = ideal_bond_lengths[aa_short_to_idx['G']].unsqueeze(0)
        gly_ideal_angles = ideal_bond_angles[aa_short_to_idx['G']].unsqueeze(0)
        gly_chi_angle_atom_indices = aa_to_chi_angle_atom_index[aa_short_to_idx['G']].unsqueeze(0)
        gly_alignment_indices = alignment_indices[aa_short_to_idx['G']].unsqueeze(0)
        gly_leftover_indices = aa_to_leftover_atoms[aa_short_to_idx['G']].unsqueeze(0)
        self.register_buffer('ideal_aa_coords', torch.cat([ideal_aa_coords, gly_ideal_coords], dim=0))
        self.register_buffer('ideal_bond_lengths', torch.cat([ideal_bond_lengths, gly_ideal_bls], dim=0))
        self.register_buffer('ideal_bond_angles', torch.cat([ideal_bond_angles, gly_ideal_angles], dim=0))
        self.register_buffer('aa_to_chi_angle_atom_index', torch.cat([aa_to_chi_angle_atom_index, gly_chi_angle_atom_indices], dim=0))

        self.register_buffer('alignment_indices', torch.cat([alignment_indices, gly_alignment_indices], dim=0))
        self.register_buffer('leftover_atom_indices', torch.cat([aa_to_leftover_atoms, gly_leftover_indices], dim=0))
        self.register_buffer('aa_to_needs_alignment', ~(self.alignment_indices == MAX_NUM_RESIDUE_ATOMS).all(dim=-1)) # type: ignore

        self.register_buffer('backbone_frame_mask_1', torch.tensor([True, True, False, True, False]))
        self.register_buffer('backbone_frame_mask_2', torch.tensor([True, True, False, True, True]))
        self.register_buffer('n_ca_cb_indices', torch.tensor([[[0], [1], [4]]]))
        self.register_buffer('n_ca_c_indices', torch.tensor([[[0], [1], [2]]]))
        self.register_buffer('frame_n_ca_c_indices', torch.tensor([[[0], [1], [3]]]))
        self.register_buffer('aa_to_chi_angle_mask', aa_to_chi_angle_mask)

        self.register_buffer('backbone_c_ca_n_atoms', torch.tensor([[[2, 2, 2], [1, 1, 1], [0, 0, 0]]]))
        self.register_buffer('backbone_nh_bond_length', torch.tensor([[1.0]]))
        self.register_buffer('backbone_nh_bond_angle', torch.tensor([120.0]))

        # Create helper tensors for building non-rotatable hydrogens on aas.
        ideal_prot_padded = F.pad(ideal_prot_aa_coords, (0, 0, 0, 1, 0, 0), 'constant', torch.nan)
        ideal_prot_padded = torch.cat([ideal_prot_padded, ideal_prot_padded[aa_short_to_idx['G']].unsqueeze(0)], dim=0)
        self.register_buffer('ideal_prot_aa_coords', ideal_prot_padded)
        self.register_buffer('aa_to_hydrogen_alignment_triad_indices', torch.cat([aa_to_hydrogen_alignment_triad_indices, aa_to_hydrogen_alignment_triad_indices[aa_short_to_idx['G']].unsqueeze(0)], dim=0))
        self.register_buffer('aa_to_hydrogen_alignment_index', torch.cat([aa_to_hydrogen_alignment_index, aa_to_hydrogen_alignment_index[aa_short_to_idx['G']].unsqueeze(0)], dim=0))

        # Create masks for residues with optional hydrogens.
        cys_hydrogen_optional_mask = torch.zeros(MAX_NUM_PROTONATED_RESIDUE_ATOMS + 1, dtype=torch.bool)
        his_hydrogen_optional_mask = torch.zeros(MAX_NUM_PROTONATED_RESIDUE_ATOMS + 1, dtype=torch.bool)
        cys_hydrogen_optional_mask[hydrogen_extended_dataset_atom_order['C'].index(optional_hydrogen_map['C'][0])] = True
        for atom in optional_hydrogen_map['H']:
            his_hydrogen_optional_mask[hydrogen_extended_dataset_atom_order['H'].index(atom)] = True
        self.register_buffer('cys_hydrogen_optional_mask', cys_hydrogen_optional_mask)
        self.register_buffer('his_hydrogen_optional_mask', his_hydrogen_optional_mask)

        # Helper tensors for methyl cap building:
        self.register_buffer('cap_nc_dist', torch.tensor([1.33484]))
        self.register_buffer('nt_cap_indices', torch.tensor([2, 1, 0]))
        self.register_buffer('nt_carbon_angle', torch.tensor([124.8122]))
        self.register_buffer('nt_cap_alignment_coords', torch.tensor([[2.542540, 2.501473, -0.877191], [1.247870, 2.778495, -1.047117], [0.190429, 1.770458, -1.245623]]))
        self.register_buffer('nt_cap_remaining_coords', torch.tensor([[2.974705, 1.360642, -0.916232], [3.463361, 3.667905, -0.600216], [0.906658, 3.713693, -0.951603], [3.176811, 4.196102, 0.274407], [3.402269, 4.350083, -1.426998], [4.492675, 3.336663, -0.482255]]))
        self.nt_cap_atom_types = ['O', 'C', 'H', 'H', 'H', 'H']
        self.register_buffer('ct_cap_indices', torch.tensor([0, 1, 2]))
        self.register_buffer('ct_nitrogen_angle', torch.tensor([118.7812]))
        self.register_buffer('ct_cap_alignment_coords', torch.tensor([[0.263855, -0.282649, 1.828291], [-0.864910, -1.324407, 1.977151], [-1.895221, -1.281854, 1.128452]]))
        self.register_buffer('ct_cap_remaining_coords', torch.tensor([[-2.958410, -2.273808, 1.041114], [-1.885658, -0.516907, 0.462938], [-3.754330, -1.878560, 0.403826], [-3.305397, -2.481747, 2.070469], [-2.607824, -3.192171, 0.625321]]))
        self.ct_cap_atom_types = ['C', 'H', 'H', 'H', 'H'] 
        final_cap_atom_types = ['C', *self.nt_cap_atom_types, 'N', *self.ct_cap_atom_types]
        self.register_buffer('final_cap_atomic_numbers', torch.tensor([atom_to_atomic_number[x] for x in final_cap_atom_types]))
        self.register_buffer('proline_mask', torch.tensor([True, True, True, False, True, True, True, *([True] * (len(self.ct_cap_atom_types) + 1))]))

        # Compute the number of chi angle bins, the embedding dimensionality, and the index to degree bin tensor.
        self.num_chi_bins = int((180 * 2) / self.chi_angle_rbf_bin_width)
        self.chi_angle_embed_dim = self.num_chi_bins * 4
        self.register_buffer('index_to_degree_bin', torch.arange(CHI_BIN_MIN, CHI_BIN_MAX, self.chi_angle_rbf_bin_width).float())

        self.register_buffer('histidine_nitrogen_indices', torch.tensor([hydrogen_extended_dataset_atom_order['H'].index('ND1'), hydrogen_extended_dataset_atom_order['H'].index('NE2')]))
        self.register_buffer('histidine_hydrogen_indices', torch.tensor([hydrogen_extended_dataset_atom_order['H'].index('HD1'), hydrogen_extended_dataset_atom_order['H'].index('HE2')]))

    def compute_binned_degree_basis_function(self, degrees: torch.Tensor, std_dev: Optional[float] = None) -> torch.Tensor:
        """
        Given degrees tensor (N, 4) of degrees between -180 and 180, computes the density of that value in circularly symmetric bin space.
            Degree values can be NaN.

        Basically implements an RBF for degrees between [-180, 180) with a standard deviation of half the bin width in degrees by 
            default in modulus 360 degrees so -180 and 179 are 1 degree apart.

        Each bin is a gaussian centered at the bin center (increments of 5 degrees) with a standard deviation of 2.5 degrees by default.
        """
        if degrees.shape[0] == 0:
            return torch.empty(0, 4, self.chi_angle_embed_dim, device=degrees.device)

        # Default standard deviation to half the bin width in degrees.
        if std_dev is None:
            std_dev = self.chi_angle_rbf_bin_width / 2
        
        # Reshape the input to allow subtraction broadcasting with the bin centers.
        degree_input_shape = degrees.shape
        if degrees.dim() != 1:
            degrees = degrees.flatten().unsqueeze(-1)

        # Compute offset in degrees relative to the current angle in circular bin space.
        bin_degrees_exp = self.index_to_degree_bin.expand(degrees.shape[0], -1) # type: ignore
        circular_bin_distance = torch.minimum(torch.remainder(bin_degrees_exp - degrees, 360), torch.remainder(degrees - bin_degrees_exp, 360))

        # Encode offset in a gaussian with standard deviation of 5 degrees by default meaning the width of 1 bin is within +/- 1 of std_dev
        A = torch.exp((-1/2) * (circular_bin_distance / std_dev) ** 2)

        # Returns encoding normalized to sum to 1
        out = A / A.sum(dim=1).unsqueeze(-1)

        return out.reshape(*degree_input_shape, -1)

    def _place_tyr_hydrogens(self, tyr_coords: torch.Tensor, sequence_indices: torch.Tensor, chi_3_angles: torch.Tensor):
        """
        Places the 'HH' hydrogen atom of a tyrosine residue based on the given coordinates, sequence indices, and chi_3 angles.
        Should be called after the rest of the atoms have been placed since the ring atoms get rotated by alignment process.

        Args:
            tyr_coords (torch.Tensor): The coordinates of the tyrosine residue (M, 15, 3)
            sequence_indices (torch.Tensor): The sequence indices of the tyrosine residue (M,)
            chi_3_angles (torch.Tensor): The chi_3 angles of the tyrosine residue. (M,)

        Returns:
            torch.Tensor: The updated coordinates of the tyrosine residue with the hydrogen atoms placed. (M, 15, 3)
        """
        # Replaces tyr hydrogens after placing the rest of the atoms. Exactly the same as _adjust_chi_rotatable_ideal_atom_placements.
        chi_number = 2
        curr_indices = self.aa_to_chi_angle_atom_index[sequence_indices] # type: ignore

        prev_coords = tyr_coords.clone().gather(1, curr_indices[:, chi_number, :3].unsqueeze(-1).expand(-1, -1, 3))
        ideal_bond_lengths = self.ideal_bond_lengths[sequence_indices][:, chi_number].unsqueeze(-1) # type: ignore
        ideal_bond_angles = torch.deg2rad(self.ideal_bond_angles[sequence_indices][:, chi_number].unsqueeze(-1)) # type: ignore
        new_chi_angles = torch.deg2rad(chi_3_angles.unsqueeze(-1))
        next_coords = extend_coordinates(prev_coords, ideal_bond_lengths, ideal_bond_angles, new_chi_angles)

        # Update the cloned ideal coordinates with the newly computed coordinate.
        tyr_coords[torch.arange(tyr_coords.shape[0], device=tyr_coords.device), curr_indices[:, chi_number, 3]] = next_coords

        return tyr_coords

    def add_nonrotatable_hydrogens(self, heavy_atom_coords: torch.Tensor, sequence_indices: torch.Tensor):
        """
        Adds non-rotatable hydrogen coordinates.
        Does not add backbone NHs, need to handle this separately since it depends on adjacent resdiue for phi/psi.
        """

        if heavy_atom_coords.shape[0] == 0:
            return torch.empty(0, MAX_NUM_PROTONATED_RESIDUE_ATOMS + 1, 3, device=heavy_atom_coords.device)

        output_coords = F.pad(heavy_atom_coords, (0, 0, 0, MAX_NUM_PROTONATED_RESIDUE_ATOMS - MAX_NUM_RESIDUE_ATOMS + 1, 0, 0), 'constant', torch.nan)

        # Create expanded ideal coordinates with hydrogen locations we need to align.
        expanded_ideal_coords = self.ideal_prot_aa_coords[sequence_indices].double() # type: ignore
        expanded_traid_indices = self.aa_to_hydrogen_alignment_triad_indices[sequence_indices] # type: ignore
        expanded_alignment_indices = self.aa_to_hydrogen_alignment_index[sequence_indices] # type: ignore

        # Loop through alignment triads and align hydrogens from ideal coordinates to heavy atom coordinates.
        for idx in range(MAX_NUM_TRIPLETS_PER_RESIDUE):
            curr_triads = expanded_traid_indices[:, idx]
            curr_hydrogen_indices = expanded_alignment_indices[:, idx]

            # Remove padding indices.
            resindex, hydr_idx_idx = (curr_hydrogen_indices != MAX_NUM_PROTONATED_RESIDUE_ATOMS).nonzero(as_tuple=True)
            hydr_idx = curr_hydrogen_indices[resindex, hydr_idx_idx]
            
            # Gather the fixed and mobile coordinates for the alignment.
            triad_indices_exp = curr_triads.unsqueeze(-1).expand(-1, -1, 3)
            fixed_triad_coords = output_coords.gather(1, triad_indices_exp)
            mobile_triad_coords = expanded_ideal_coords.gather(1, triad_indices_exp)

            # Gather the hydrogen coordinates that we will align.
            ideal_hydrogen_coords = expanded_ideal_coords[resindex].gather(1, hydr_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3))
            hydrogen_alignment_matrices = compute_alignment_matrices(fixed_triad_coords[resindex], mobile_triad_coords[resindex])
            aligned_hydrogen_coords = apply_transformation(ideal_hydrogen_coords, *hydrogen_alignment_matrices).squeeze(1)
            assert (~aligned_hydrogen_coords.isnan()).any().cpu().item() or aligned_hydrogen_coords.numel() == 0, "Failed to align non-rotatable hydrogens..."

            # Update the output coordinates with the aligned hydrogens.
            output_coords[resindex, hydr_idx] = aligned_hydrogen_coords

        return output_coords

    def _compute_methyl_cap_coordinates(self, phi_psi: torch.Tensor, curr_coords: torch.Tensor, seq_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the coordinates of a methylamine cap for N-terminal and C-terminal caps.

            - NOTE: because I'm rigidly aligning the hydrogens on the methyl cap without computing a hydrogen 'chi-angle', 
                the cap hydrogens aren't guaranteed to be staggared.

            - NOTE: technically we don't need to compute coordinates since we could pull the carbon coordinates 
                from adjacent residues, but doing it this way allows easily randomizing phi/psi for residues next to chain breaks.

        """

        # Handle NaN phi/psi angles by randomly sampling them.
        if phi_psi[0].isnan().item():
            phi_psi[0] = (torch.rand_like(phi_psi[0]) * 360) - 180
        if phi_psi[1].isnan().item():
            phi_psi[1] = (torch.rand_like(phi_psi[1]) * 360) - 180

        # Unpack phi and psi angles.
        phi, psi = torch.deg2rad(phi_psi)

        # Handle phi:
        #   gathers [C_{0}, CA_{0}, N_{0}] coords.
        alignment_coords = curr_coords.gather(0, self.nt_cap_indices.unsqueeze(-1).expand(-1, 3)).unsqueeze(0) # type: ignore
        imputed_carbon = extend_coordinates(alignment_coords, self.cap_nc_dist.unsqueeze(-1), torch.deg2rad(self.nt_carbon_angle.unsqueeze(-1)), phi.reshape(1, 1)) # type: ignore
        #   order is [C_{-1}, N_{0}, CA_{0}]
        fixed_coords = torch.cat([imputed_carbon, alignment_coords[:, 2], alignment_coords[:, 1]], dim=0)
        mobile_coords = self.nt_cap_alignment_coords.unsqueeze(0) # type: ignore
        nt_aligned_cap_coords = apply_transformation(self.nt_cap_remaining_coords.unsqueeze(0), *compute_alignment_matrices(fixed_coords.unsqueeze(0), mobile_coords)).squeeze(0) # type: ignore
        nt_cap_coords = torch.cat([imputed_carbon, nt_aligned_cap_coords], dim=0)

        # Handle psi:
        #   Gathers N_{0}, CA_{0}, C_{0} coords
        alignment_coords = curr_coords.gather(0, self.ct_cap_indices.unsqueeze(-1).expand(-1, 3)).unsqueeze(0) # type: ignore
        imputed_nitrogen = extend_coordinates(alignment_coords, self.cap_nc_dist.unsqueeze(-1), torch.deg2rad(self.ct_nitrogen_angle.unsqueeze(-1)), psi.reshape(1, 1)) # type: ignore
        #   Order is [CA_{0}, C_{0}, N_{+1}]
        fixed_coords = torch.cat([alignment_coords[:,1], alignment_coords[:, 2], imputed_nitrogen])
        mobile_coords = self.ct_cap_alignment_coords.unsqueeze(0) # type: ignore
        ct_aligned_cap_coords = apply_transformation(self.ct_cap_remaining_coords.unsqueeze(0), *compute_alignment_matrices(fixed_coords.unsqueeze(0), mobile_coords)).squeeze(0) # type: ignore
        ct_cap_coords = torch.cat([imputed_nitrogen, ct_aligned_cap_coords], dim=0)

        all_cap_coords = torch.cat([nt_cap_coords, ct_cap_coords], dim=0)
        final_cap_atom_types = self.final_cap_atomic_numbers # type: ignore
        assert isinstance(final_cap_atom_types, torch.Tensor) # appease type checker
        if seq_idx == aa_short_to_idx['P']:
            all_cap_coords = all_cap_coords[self.proline_mask] # type: ignore
            final_cap_atom_types = final_cap_atom_types[self.proline_mask] # type: ignore

        return all_cap_coords, final_cap_atom_types 

    def _generate_ideal_coord_rotamers(self, sequence_indices, chi_angles) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rotates the ideal amino acid coordinates to desired chi angles.

        Args:
            sequence_indices (torch.Tensor): The sequence indices. (N,) elements in range [0, 20]
            chi_angles (torch.Tensor): The chi angles. (N, 4) padded with NaN

        Returns:
            torch.Tensor: The ideal amino acid coordinates in the desired rotamer conformation. (N, MAX_NUM_RESIDUE_ATOMS, 3)
            torch.Tensor: The unadjusted ideal amino acid coordinates. (N, MAX_NUM_RESIDUE_ATOMS, 3)
        """

        # Create ideal atomic coordiantes for each residue and make a copy which we will use to compute alignment.
        ideal_coords = self.ideal_aa_coords[sequence_indices].double() # type: ignore
        unadjusted_ideal_coords = ideal_coords.clone()

        # Separate the indices into the atoms that will be used to compute the next atom location and the next atom index.
        chi_indices = self.aa_to_chi_angle_atom_index[sequence_indices] # type: ignore
        extension_indices = chi_indices[:, :, :3]
        next_index = chi_indices[:, :, 3]

        all_bond_lengths = self.ideal_bond_lengths[sequence_indices] # type: ignore
        all_bond_angles = torch.deg2rad(self.ideal_bond_angles[sequence_indices]) # type: ignore
        all_chi_angles = torch.deg2rad(chi_angles)

        for chi_i in range(4):
            # Pull the atom indices necessary for computing the current next atom location.
            # Index the ideal amino acid coordinates with curr_indices to get the current atom locations.
            curr_indices = extension_indices[:, chi_i]
            residue_idces = (~chi_angles[:, chi_i].isnan()).nonzero().flatten()

            # (N x 3 x 3) coordinates of the ideal residue that we will use to compute the next atom location.
            prev_atoms = ideal_coords[residue_idces].gather(1, curr_indices[residue_idces].unsqueeze(-1).expand(-1, -1, 3))

            # Get ideal bond lengths, bond angles and dihedrals for current chi angle index:
            ideal_bond_lengths = all_bond_lengths[residue_idces, chi_i].unsqueeze(-1) # type: ignore
            ideal_bond_angles = all_bond_angles[residue_idces, chi_i].unsqueeze(-1) # type: ignore
            target_chi_angles = all_chi_angles[residue_idces, chi_i].unsqueeze(-1)

            # Compute the next atom location.
            next_coords = extend_coordinates(prev_atoms, ideal_bond_lengths, ideal_bond_angles, target_chi_angles)
            ideal_coords[residue_idces, next_index[residue_idces, chi_i]] = next_coords
            assert (~next_coords.isnan()).any().cpu().item() or next_coords.numel() == 0, "Failed to converge chi angle extension..."

        return ideal_coords, unadjusted_ideal_coords
    
    def build_rotamers(
        self, 
        backbone_coords: torch.Tensor, 
        chi_angles: torch.Tensor, 
        sequence_indices: torch.Tensor, 
        backbone_alignment_matrices: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        add_nonrotatable_hydrogens: bool = False,
        return_backbone_alignment_matrices: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Builds full atom coordinates given backbone coordinates, sequence labels, and chi angles.
        First rotates and aligns ideal amino acid atoms to the desired rotamer.
        Then, aligns adjusted ideal amino acids to the backbone coordinates.
            
        Args:
            backbone_coords (torch.Tensor): Tensor containing backbone coordinates. (N, 5, 3) 
                where middle index is in order (N, CA, CB, C, O)
            chi_angles (torch.Tensor): Tensor containing chi angles. (N, 4)
            sequence_indices (torch.Tensor): Tensor containing sequence labels. (N,) where each element is in range [0, 19]

            backbone_alignment_matrices (Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): Optional 
                tuple containing backbone alignment matrices, computed if not available.
        
        Returns:
            torch.Tensor: Tensor containing the full atom coordinates. (N, MAX_NUM_RESIDUE_ATOMS, 3)
            or 
            a tuple of the full atom coordinates and the backbone alignment matrices if return_backbone_alignment_matrices is True.
        """

        # Modifies coords contain the (partially, TYR is delayed to after final alignment) rotated coordinates.
        ideal_coords, ideal_coords_unadjusted = self._generate_ideal_coord_rotamers(sequence_indices, chi_angles)

        # Lookup the indices of the residues that need to be aligned.
        residue_needs_alignment = self.aa_to_needs_alignment[sequence_indices] # type: ignore
        # Lookup precomputed indices with which to compute a rigid body alignment between the adjusted ideal coordinates and leftover atoms.
        leftover_atom_alignment_indices = self.alignment_indices[sequence_indices[residue_needs_alignment]] # type: ignore
        # Lookup the indices of the atoms that we will overwrite the un-adjusted ideal coordinates with.
        leftover_mobile_indices = self.leftover_atom_indices[sequence_indices[residue_needs_alignment]] # type: ignore

        # Gather the fixed and mobile coordinates for the alignment.
        expanded_alignment_indices = leftover_atom_alignment_indices.unsqueeze(-1).expand(-1, -1, 3)
        fixed_coords = ideal_coords[residue_needs_alignment].gather(1, expanded_alignment_indices)
        mobile_coords = ideal_coords_unadjusted[residue_needs_alignment].gather(1, expanded_alignment_indices) # type: ignore

        # Gather the leftover coordinates that don't get updated by the _adjust_chi_rotatable_ideal_atom_placements function.
        residue_idx, leftover_idx = (leftover_mobile_indices != MAX_NUM_RESIDUE_ATOMS).nonzero(as_tuple=True)
        leftover_coord_idces = leftover_mobile_indices[residue_idx, leftover_idx]
        leftover_mobile_coords = ideal_coords_unadjusted[residue_needs_alignment][residue_idx, leftover_coord_idces]

        # Align the adjusted ideal coordinates to the backbone coordinates.
        leftover_alignment_matrices = compute_alignment_matrices(fixed_coords, mobile_coords)
        leftover_mobile_coords = apply_transformation(leftover_mobile_coords.unsqueeze(1), *[x[residue_idx] for x in leftover_alignment_matrices]).squeeze(1)

        # Update the adjusted ideal coordinates with the leftover mobile coordinates.
        subset = ideal_coords[residue_needs_alignment]
        subset[residue_idx, leftover_coord_idces] = leftover_mobile_coords
        ideal_coords[residue_needs_alignment] = subset

        # Run the extension algorithm using aligned residues to adjust the hydrogens placed on tyrosine.
        is_tyr_mask = sequence_indices == aa_short_to_idx['Y']
        tyr_coords = self._place_tyr_hydrogens(ideal_coords[is_tyr_mask], sequence_indices[is_tyr_mask], chi_angles[is_tyr_mask, 2])
        ideal_coords[is_tyr_mask] = tyr_coords

        # Align the (N, Ca, Cb) completely transformed ideal coordinates to the backbone coordinates.
        not_gly_mask = (sequence_indices != aa_short_to_idx['G']) & (sequence_indices != aa_short_to_idx['X'])

        # Align the completely transformed ideal coordinates to the backbone coordinates.
        fixed_backbone_coords = backbone_coords.clone().gather(1, torch.where(
            not_gly_mask.view(-1, 1, 1), 
            self.n_ca_c_indices.expand(ideal_coords.shape[0], 3, 3), # type: ignore in this case the indices 0, 1, 2 are N, CA, CB
            self.frame_n_ca_c_indices.expand(ideal_coords.shape[0], 3, 3) # type: ignore
        ))
        mobile_backbone_coords = ideal_coords.clone().gather(1, torch.where(
            not_gly_mask.view(-1, 1, 1),
            self.n_ca_cb_indices.expand(ideal_coords.shape[0], 3, 3), # type: ignore
            self.n_ca_c_indices.expand(ideal_coords.shape[0], 3, 3), # type: ignore
        )) 

        if backbone_alignment_matrices is None:
            backbone_alignment_matrices = compute_alignment_matrices(fixed_backbone_coords.double(), mobile_backbone_coords)

        # Overwrite the aligned backbone coordinates with the true backbone coordinates to avoid only using ideal phi/psi angles.
        output_residx, output_coord_idx = (~ideal_coords[:, :, 0].isnan()).nonzero(as_tuple=True)
        ideal_coords[output_residx, output_coord_idx] = apply_transformation(ideal_coords[output_residx, output_coord_idx].unsqueeze(1), *[x[output_residx] for x in backbone_alignment_matrices]).squeeze(1)
        ideal_coords[:, :4] = backbone_coords[:, self.backbone_frame_mask_2]

        if add_nonrotatable_hydrogens:
            ideal_coords = self.add_nonrotatable_hydrogens(ideal_coords, sequence_indices)

        if not return_backbone_alignment_matrices:
            return ideal_coords
        else:
            return ideal_coords, backbone_alignment_matrices
    
    def impute_backbone_nh_coords(self, full_atom_coords: torch.Tensor, residue_sequence_indices: torch.Tensor, phi_angles: torch.Tensor) -> torch.Tensor:
        """
        Computes the coordinates of the backbone hydrogen atoms given the full atom coordinates and the phi angles.
        """

        assert full_atom_coords.shape[1] > 5, 'You passed the backbone coordinates not the fa_coordinates to impute_backbone_nh_coords'

        # Gather the C, Ca, and N atoms.
        backbone_atoms = full_atom_coords.gather(1, self.backbone_c_ca_n_atoms.expand(full_atom_coords.shape[0], -1, -1)) # type: ignore
        nh_bond_length = self.backbone_nh_bond_length.expand(full_atom_coords.shape[0], -1) # type: ignore
        bond_angle = self.backbone_nh_bond_angle.expand(full_atom_coords.shape[0], -1) # type: ignore
        dihedral_angle = phi_angles + 180

        # Compute the coordinates of the NH atoms.
        nh_coords = extend_coordinates(backbone_atoms, nh_bond_length, bond_angle.deg2rad(), dihedral_angle.deg2rad())

        # Output the NH coordinates for non-proline residues.
        output = torch.full((full_atom_coords.shape[0], 3), torch.nan, device=full_atom_coords.device)
        pro_mask = residue_sequence_indices == aa_short_to_idx['P']
        output[~pro_mask] = nh_coords[~pro_mask]

        return output
    
    def compute_ligand_donor_hydrogens(self, ligand_coords: torch.Tensor, lig_atomic_numbers: torch.Tensor, lig_lig_eidx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def cleanup_titratable_hydrogens(
            self, fa_coords: torch.Tensor, sequence_indices: torch.Tensor, bb_nh_coords: torch.Tensor, batch_data,
            hbond_network_detector: nn.Module, covalent_cys_bond_length: float = 2.5, histidine_max_hbond_distance: float = 2.5
    ) -> torch.Tensor:

        sequence_indices_ = sequence_indices.clone()
        sequence_indices_[sequence_indices == aa_short_to_idx['X']] = aa_short_to_idx['G']

        # Handle CYS residues first.
        ### Identify all cysteine residues.
        cys_mask = sequence_indices_ == aa_short_to_idx['C']
        cys_s_coords = fa_coords[cys_mask, hydrogen_extended_dataset_atom_order['C'].index('SG')]
        cys_mask_indices = cys_mask.nonzero().flatten()
        ### Identify all cysteine residues within covalent bond distance of each other and remove the HG hydrogens.
        within_distance_mask = torch.cdist(cys_s_coords, cys_s_coords) < covalent_cys_bond_length
        same_batch_mask = batch_data.batch_indices[cys_mask_indices].unsqueeze(-1) == batch_data.batch_indices[cys_mask_indices].unsqueeze(0)
        within_distance_mask = within_distance_mask & (~torch.eye(within_distance_mask.shape[0], device=within_distance_mask.device, dtype=torch.bool)) & same_batch_mask
        indices_to_update = cys_mask_indices[within_distance_mask.any(dim=-1)]
        fa_coords[indices_to_update, hydrogen_extended_dataset_atom_order['C'].index('HG')] = torch.nan

        # raise NotImplementedError

        # Handle HIS residues.
        # TODO: could add back in Donor-H-Acceptor angle constraints for h-bonding, but this is hacky anyway.
        his_mask = sequence_indices_ == aa_short_to_idx['H']
        his_mask_indices = his_mask.nonzero().flatten()

        ## CASE 1: N of His is within 2.5A of a metal ion, deprotonate it.
        #######################

        ## CASE 2: N of His is within 2.5A of polar hydrogen, deprotonate it.
        #######################
        ### Case 2.1: HB-Donor atom on a residue sidechain:
        #### Collect all the sidechain donor hydrogen coordinates
        # row_indices, col_indices = (~fa_coords.isnan().any(dim=-1)).nonzero(as_tuple=True)
        # all_sc_donorh_coords = fa_coords[row_indices].gather(1, hbond_network_detector.hbond_donor_aa_idx_to_donor_heavy_pair[sequence_indices_[row_indices], col_indices].unsqueeze(-1).expand(-1, -1, 3)).flatten(end_dim=-2) # type: ignore
        # #### Track which residue each donor hydrogen belongs to.
        # row_indices_exp = row_indices.repeat_interleave(hbond_network_detector.hbond_donor_aa_idx_to_donor_heavy_pair.shape[-1]) # type: ignore
        # mask = ~all_sc_donorh_coords.isnan().any(dim=-1)
        # all_sc_donorh_coords = all_sc_donorh_coords[mask]
        # row_indices_exp = row_indices_exp[mask]
        # batch_indices_exp = batch_data.batch_indices[row_indices_exp]

        # #### Compute all histidine nitrogen coordinates and indices.
        # nitrogen_indices_exp = self.histidine_nitrogen_indices.unsqueeze(0).unsqueeze(-1).expand(his_mask_indices.shape[0], 2, 3) # type: ignore
        # his_hydrogen_indices_exp = self.histidine_hydrogen_indices.unsqueeze(0).expand(his_mask_indices.shape[0], -1) # type: ignore

        # his_mask_indices_exp = his_mask_indices.repeat_interleave(2)
        # histidine_batch_indices = batch_data.batch_indices[his_mask_indices_exp]
        # histidine_nitrogen_coords = fa_coords[his_mask_indices].gather(1, nitrogen_indices_exp).flatten(end_dim=-2)

        # #### Compute distances between donors and acceptors and figure out if the hydrogens are within h-bonding distance (and not from the same residue).
        # distances = torch.cdist(all_sc_donorh_coords, histidine_nitrogen_coords)
        # not_same_index_mask = row_indices_exp.unsqueeze(-1) != his_mask_indices_exp.unsqueeze(0) # Mimics the output of cdist with broadcasting.
        # is_same_batch_idx_mask = batch_indices_exp.unsqueeze(-1) == histidine_batch_indices.unsqueeze(0)
        # putative_hbond_acceptor_nitrogen_indices = ((distances < histidine_max_hbond_distance) & not_same_index_mask & is_same_batch_idx_mask).any(dim=0).nonzero().flatten()

        # sc_donor_his_mask_indices_to_modify = his_mask_indices_exp.flatten()[putative_hbond_acceptor_nitrogen_indices]
        # sc_donor_his_nitrogen_indices_to_modify = his_hydrogen_indices_exp.flatten()[putative_hbond_acceptor_nitrogen_indices]

        # ### Case 2.2: HB-Donor atom is bb NH:
        # #### Compute distances between backbone NH and histidine nitrogen atoms.
        # distances = torch.cdist(bb_nh_coords, histidine_nitrogen_coords)
        # putative_hbond_acceptor_nitrogen_indices_bb = ((distances < histidine_max_hbond_distance) & (batch_data.batch_indices.unsqueeze(-1) == histidine_batch_indices.unsqueeze(0))).any(dim=0).nonzero().flatten()
        # bb_donor_his_mask_indices_to_modify = his_mask_indices_exp.flatten()[putative_hbond_acceptor_nitrogen_indices_bb]
        # bb_donor_his_nitrogen_indices_to_modify = his_hydrogen_indices_exp.flatten()[putative_hbond_acceptor_nitrogen_indices_bb]

        # ### Case 2.3: HB-Donor atom is on a ligand:
        # self.compute_ligand_donor_hydrogens(batch_data.ligand_data.lig_coords, batch_data.ligand_data.ligand_atomic_numbers, batch_data.ligand_data.lig_lig_edge_index)
        # raise NotImplementedError

        ### Case 3: TODO: N of His (not previously marked for deprotonation) is within 3.5 of an acceptor atom and should be protonated.
        #######################

        ### Case 4: TODO: Default all unmodified histidine nitrogen atoms to epsilon tautomer.
        #######################
        ### Default: Keep epsilon hydrogen only
        fa_coords[his_mask_indices, hydrogen_extended_dataset_atom_order['H'].index('HD1')] = torch.nan

        ######################
        #### Perform all histidine updates.
        ### TODO: If two histidines are interacting, we should deprotonate the one with the fewest deprotonated nitrogens or just the first in resnum order for now...
        # fa_coords[sc_donor_his_mask_indices_to_modify, sc_donor_his_nitrogen_indices_to_modify] = torch.nan
        # fa_coords[bb_donor_his_mask_indices_to_modify, bb_donor_his_nitrogen_indices_to_modify] = torch.nan

        return fa_coords


def compute_alignment_matrices(fixed: torch.Tensor, mobile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the rotation and translation matrices that align the mobile coordinates to the fixed coordinates.
    Returns: the rotation matrix, the center of mass of the mobile coordinates, and the center of mass of the fixed coordinates.
    """
    # Compute center of mass of fixed and mobile coordinate lists.
    fixed_coords_com = fixed.mean(dim=1, keepdim=True)
    mob_coords_com = mobile.mean(dim=1, keepdim=True)

    # Center the fixed and mobile coordinate lists.
    mob_coords_cen = mobile - mob_coords_com
    targ_coords_cen = fixed - fixed_coords_com

    if mob_coords_cen.isnan().any() or targ_coords_cen.isnan().any():
        raise ValueError("NaNs in alignment matrices...")

    # Compute the transformation that minimizes the RMSD between the fixed and mobile coordinate lists.
    C = mob_coords_cen.transpose(1, 2) @ targ_coords_cen
    U, _, Wt = torch.linalg.svd(C)
    R = U @ Wt
    neg_det_mask = torch.linalg.det(R) < 0.0
    Wt_neg_det = Wt.clone() # Avoids in-place modification of Wt.
    Wt_neg_det[neg_det_mask, -1] *= -1
    R[neg_det_mask] = U[neg_det_mask] @ Wt_neg_det[neg_det_mask]

    return R, mob_coords_com, fixed_coords_com


def apply_transformation(coords: torch.Tensor, R: torch.Tensor, mob_coords_com: torch.Tensor, fixed_coords_com: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation and translation matrices computed in compute_alignment_matrices to coords.
    """
    return ((coords - mob_coords_com) @ R) + fixed_coords_com


def extend_coordinates(prev_atom_coords: torch.Tensor, bond_lengths: torch.Tensor, bond_angles: torch.Tensor, dihedral_angles: torch.Tensor) -> torch.Tensor:
    """
    Extends the coordinates of a molecule based on bond lengths, bond angles, and dihedral angles.

    Args:
        coords (torch.Tensor): A (N, 3, 3) coodinate tensor.
        bond_lengths (torch.Tensor): (N, 1) The ideal length of bond of the atom to be added.
        bond_angles (torch.Tensor): (N, 1) The ideal bond angle of atom being added.
        dihedral_angles (torch.Tensor): (N, 1) The dihedral angles being added.

    Returns:
        torch.Tensor: The (N, 3) coordinates of the fourth atom defining the desired dihedral 
            angle with the ideal bond length and bond angle.
    """

    # Small epsilon to avoid division by zero.
    eps = torch.tensor(1e-6, device=prev_atom_coords.device)

    bc = prev_atom_coords[:, 1] - prev_atom_coords[:, 2]
    bc = bc / (torch.linalg.vector_norm(bc, dim=-1, keepdim=True) + eps)
    ba = torch.cross(prev_atom_coords[:, 1] - prev_atom_coords[:, 0], bc, dim=1)
    ba = ba / (torch.linalg.vector_norm(ba, dim=-1, keepdim=True) + eps)
    m1 = torch.cross(ba, bc, dim=1)
    d1 = bond_lengths * torch.cos(bond_angles)
    d2 = bond_lengths * torch.sin(bond_angles) * torch.cos(dihedral_angles)
    d3 = -1 * bond_lengths * torch.sin(bond_angles) * torch.sin(dihedral_angles)
    next_coords = prev_atom_coords[:, 2] + (bc * d1) + (m1 * d2) + (ba * d3)
    return next_coords


@torch.no_grad()
def compute_chi_angle_accuracies(sampled_chi_angles, ground_truth_chi_angles, rotamer_builder):
    """
    Given N x 4 tensors of sampled and ground truth chi angles, computes the accuracy of each chi angle predictions.
        NOTE: Accuracy is cumulative, meaning chi_2_acc is computed for residues for which chi-1 was predicted correctly.
    """

    reindex_tensor = torch.tensor([0, 4], dtype=torch.long, device=sampled_chi_angles.device)

    nan_mask = ground_truth_chi_angles.isnan()
    sampled_chi_angles[nan_mask] = torch.nan

    # Find angles within bin width of ground truth chi angle.
    bool_mask = torch.minimum(torch.remainder(ground_truth_chi_angles - sampled_chi_angles, 360), torch.remainder(sampled_chi_angles - ground_truth_chi_angles, 360)) < rotamer_builder.chi_angle_rbf_bin_width

    # Get the index of the first False value in each row. 
    min_indices = bool_mask.long().argmin(dim=1)

    # Handle case where all values are True or all values are False.
    first_index_mask = min_indices == 0

    # Get values for the case where min_indices is 0.
    min_values = bool_mask[first_index_mask].gather(1, min_indices[first_index_mask].unsqueeze(-1)).long()
    min_indices[first_index_mask] = reindex_tensor[min_values].flatten()

    output = {}
    for i in range(1, 5):
        # Count number of sampled chi angles that have chi_i correct.
        chi_i_correct = 0
        for j in range(i, 5):
            chi_i_correct += (min_indices == j).sum().item()

        # Count number of residues that have a chi_i.
        num_chi_i = (~(ground_truth_chi_angles[:, i - 1].isnan())).sum().item()

        # Compute accuracy for residues that have exactly i chi angles.
        chi_i_acc = chi_i_correct / max(num_chi_i, 1)
        output[f'chi_{i}_acc'] = chi_i_acc

    return output