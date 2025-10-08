from enum import unique
from ase import Atoms
from ase.units import Hartree
import torch

from e3nn.util.jit import compile_mode
from ase.data import covalent_radii
from typing import Dict, Tuple

import torch
from scipy.constants import e, epsilon_0, pi
from e3nn import o3
import scipy.special
from typing import List
import numpy as np

from graph_longrange.utils import to_dense_batch
from graph_longrange.kspace import compute_k_vectors
from scipy.constants import e, epsilon_0, pi
from e3nn import o3
from graph_longrange.utils import permute_to_e3nn_convention,to_dense_batch
import scipy.special
from graph_longrange.torchspline import (
    natural_cubic_spline_coeffs,
    NaturalCubicSplineBlock,
)
from graph_longrange.gto_hardness import (
    HardnessMatrix,
)

from graph_longrange.utils import (
    to_dense_batch,
)

from graph_longrange.gto_electrostatics import (
    gto_basis_kspace_cutoff,
)
from mace.tools.scatter import scatter_sum

# internal units are (electon, Volt, Angstrom)
# 5.526349406 * 1e-3 is the value of epsilon_0 in (electon, Volt, Angstrom) units. see docs.
FIELD_CONSTANT = 1 / (5.526349406 * 1e-3)


@compile_mode("script")
class ChargeEquilibrationBlock(torch.nn.Module):
    def __init__(self, scale_atsize: float):
        super().__init__()
        self.hardness_matrix = HardnessMatrix(scale_atsize)

    def forward(
        self,
        data: Dict[str,torch.Tensor],
        eneg: torch.Tensor,
        atomic_numbers) -> Tuple[torch.Tensor, torch.Tensor]:
        device = eneg.device
        
        total_charge = -1*data["total_charge"].unsqueeze(1)
        eneg_dense, eneg_mask = to_dense_batch(eneg.squeeze(), data["batch"])
        A_elec, A_qeq = self.hardness_matrix(data["positions"], data["batch"], atomic_numbers)
        
        masked_enegs = torch.where(eneg_mask, eneg_dense, torch.zeros_like(eneg_dense))
        masked_enegs_total_charge = torch.cat((masked_enegs, total_charge), dim=1)
        # # This caused outputs to be scrambeled for some reason during the integration into gromacs. Don't use with gromacs.
        # masked_enegs_total_charge = -1*masked_enegs_total_charge
        # dense_charges = torch.linalg.solve(new_Abar, -1*masked_enegs_total_charge)
        masked_enegs_total_charge = -1*masked_enegs_total_charge.unsqueeze(2)
        LU, pivots = torch.linalg.lu_factor(A_qeq)
        dense_charges = torch.linalg.lu_solve(LU, pivots, masked_enegs_total_charge).squeeze(2)

        dense_charges = dense_charges[:,:-1]
        output_partial_charges = dense_charges.flatten()
        eneg_mask_flatten = eneg_mask.flatten()
        output_partial_charges = output_partial_charges[eneg_mask_flatten]

        # Compute the electrostatic energy
        charge_matrix = dense_charges.unsqueeze(1) * dense_charges.unsqueeze(2)
        e_elec_matrix = torch.triu(A_elec)*charge_matrix
        e_elec = torch.sum(e_elec_matrix, dim=(1, 2))

        return output_partial_charges, e_elec
