from enum import unique
from ase import Atoms
import torch

from e3nn.util.jit import compile_mode
from ase.data import covalent_radii
from typing import Dict

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
from graph_longrange.utils import permute_to_e3nn_convention
import scipy.special
from graph_longrange.torchspline import (
    natural_cubic_spline_coeffs,
    NaturalCubicSplineBlock,
)

from mace.tools.scatter import scatter_sum

# internal units are (electon, Volt, Angstrom)
# 5.526349406 * 1e-3 is the value of epsilon_0 in (electon, Volt, Angstrom) units. see docs.
FIELD_CONSTANT = 1 / (5.526349406 * 1e-3)


@compile_mode("script")
class ChargeEquilibrationBlock(torch.nn.Module):
    def __init__(self, num_elements: int):
        super().__init__()

        self.hardness = torch.nn.Parameter(
            torch.rand(
                num_elements,
            )
        )

    def forward(
        self,
        eneg: torch.Tensor,
        data: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        device = eneg.device
        sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=device))
        sqrt_2 = torch.sqrt(torch.tensor(2, device=device))

        # count the number of unique entries in batch
        positions = data["positions"]
        batch_indices = data["batch"]
        unique_batch_indices = torch.unique(batch_indices)
        # atomic numbers is an unrolled list of atomic numbers for each molecule
        cov_radii = torch.tensor(covalent_radii, device=device)

        output_partial_charges = []
        for batch_index in unique_batch_indices:
            molecule_indices = torch.where(batch_indices == batch_index)
            mol_positions = positions[molecule_indices]
            n_atoms = len(mol_positions)
            total_charge = data["total_charge"][batch_index]
            mol_atomic_numbers = atomic_numbers[molecule_indices]

            atomic_indices = torch.argmax(data["node_attrs"][molecule_indices], dim=1)
            hardness_vals = self.hardness[atomic_indices]
            diag_vals = hardness_vals + (1 / (sqrt_pi * cov_radii[mol_atomic_numbers]))
            # compute distances between each pair of atoms
            # this only works for nonperiodic systems, we need to obey the minimum image convention for periodic systems

            # extract the [n_atoms, n_atoms] matrix of distances between each atom, accounting for periodic boundary conditions
            distances = torch.cdist(mol_positions, mol_positions)
            sigma = torch.square(cov_radii[mol_atomic_numbers]).view(-1, 1)
            gamma = torch.sqrt(sigma + sigma.t())
            A = torch.erf(distances / (sqrt_2 * gamma)) / distances
            A[torch.arange(n_atoms), torch.arange(n_atoms)] = diag_vals
            # solve the linear system of equations using to constraint that the sum of the charges is equal to the total charge with lagrange multipliers
            A = torch.cat((A, torch.ones((n_atoms, 1), device=positions.device)), dim=1)
            A = torch.cat(
                (A, torch.ones((1, n_atoms + 1), device=positions.device)), dim=0
            )

            b = -1 * eneg[molecule_indices]
            b = torch.cat(
                (b, torch.tensor([[total_charge]], device=positions.device)), dim=0
            )

            x = torch.linalg.solve(A, b)
            x = x[:-1].squeeze()
            output_partial_charges.append(x)
            # print(x)
            # check sum of charges is equal to total charge
            # print(float(torch.sum(x)), float(total_charge))

            # check sum of charges is equal to total charge
            # print(float(torch.sum(x)), float(total_charge))

        # check the sum along the atom axiz is equal to the total charge
        # print(torch.sum(output_partial_charges, dim=1))
        return torch.cat(output_partial_charges, dim=0)


@compile_mode("script")
class LocalChargeEquilibrationBlock(torch.nn.Module):
    def __init__(
        self, num_elements: int, r_cut: int = 5, covalent_radii=covalent_radii
    ):
        super().__init__()

        self.hardness = torch.nn.Parameter(
            torch.rand(
                num_elements,
            )
        )
        self.register_buffer("r_cut", torch.tensor(r_cut))
        self.register_buffer("covalent_radii", torch.tensor(covalent_radii))

    def forward(
        self,
        T: torch.Tensor,
        eneg: torch.Tensor,
        p: torch.Tensor,
        data: Dict[str, torch.Tensor],
        atomic_numbers: torch.Tensor,
    ):
        # here we solve the linear system for each molecule in the batch.
        device = p.device
        sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=device))
        sqrt_2 = torch.sqrt(torch.tensor(2, device=device))

        # count the number of unique entries in batch
        positions = data["positions"]

        # turn the positions matrix into a block diagonal matrix with each block corresponding to a molecule

        # given the distances and the cell, construct the distance matrix for each molecule in the batch.  We will index this by the edge indices, so ideally will be a block diagonal matrix

        distances = data["distances"]
        # this is a tensor of shape [n_atoms_per_batch, max_n_atoms_in_molecule], where the difference between the atoms in the molecule and the max number of atoms in the molecule is padded with zeros, to enable efficient creation of tbe batch
        batch_indices = data["batch"]

        unique_batch_indices = torch.unique(batch_indices)
        n_atoms_per_mol = []
        for i in unique_batch_indices:
            mol_indices = torch.where(batch_indices == i)[0]
            n_atoms_per_mol.append(len(mol_indices))

        # now slice the positions matrix every n_atoms_per_mol to get a list of molecule positions
        all_distances = torch.split(data["distances"], n_atoms_per_mol)
        # now slice each tensor in slice_positions to remove the extra zero padding
        all_distances = [
            x[:, :n_atoms] for x, n_atoms in zip(all_distances, n_atoms_per_mol)
        ]

        # now we have a block diagonal tensor that contains the interatomic distances in each config, accounting for periodic boundary conditions
        all_distances_tensor = torch.block_diag(*all_distances)

        # atomic numbers is an unrolled list of atomic numbers for each molecule
        cov_radii = self.covalent_radii.to(device)

        output_partial_charges = []
        for batch_index in unique_batch_indices:
            # get the indices of the ith molecule in the batch from the contiguous list of molecules
            molecule_indices = torch.where(batch_indices == batch_index)
            mol_positions = positions[molecule_indices]
            n_atoms = len(mol_positions)
            total_charge = data["total_charge"][batch_index]
            mol_atomic_numbers = atomic_numbers[molecule_indices]
            # get the individual tensor
            mol_distances = all_distances[batch_index]
            # slice the edge_index list to get the edges corresponding to this molecule

            # get edges from the connectivity tensor corresponding to this molecule
            molecule_edge_indices = torch.where((T[molecule_indices] != 0).any(dim=0))[
                0
            ]

            # slice edge_indices
            mol_edge_indices = data["edge_index"][:, molecule_edge_indices]
            n_mol_edges = mol_edge_indices.shape[1]
            # slice the T matrix to get the connectivity for that molecule
            T_mol = T[molecule_indices][:, molecule_edge_indices]

            atomic_indices = torch.argmax(data["node_attrs"][molecule_indices], dim=1)
            hardness_vals = self.hardness[atomic_indices]
            diag_vals = hardness_vals + (1 / (sqrt_pi * cov_radii[mol_atomic_numbers]))
            # compute distances between each pair of atoms
            # distances = torch.cdist(mol_positions, mol_positions)
            # now get a vector of bond lengths for each edge in the molecule by indexing the distances matrix with the edge indices
            edge_lengths = all_distances_tensor[
                mol_edge_indices[0], mol_edge_indices[1]
            ]

            # construct the A matrix in a similar way to above
            A = torch.zeros((n_atoms, n_atoms), device=positions.device)

            sigma = torch.square(cov_radii[mol_atomic_numbers]).view(-1, 1)
            gamma = torch.sqrt(sigma + sigma.t())

            # this is the H+J part
            A = torch.erf(mol_distances / (sqrt_2 * gamma)) / mol_distances
            A[torch.arange(n_atoms), torch.arange(n_atoms)] = diag_vals

            A = T_mol.t() @ A @ T_mol

            # add the C matrix, size of C is the same as T
            C = torch.zeros(
                (n_mol_edges, n_mol_edges),
                device=positions.device,
                dtype=positions.dtype,
            )

            cutoffs = torch.cos(torch.pi * edge_lengths / (2 * self.r_cut)) ** -1 - 1

            # create a diagonal matrix with cutoffs on the diagonal
            C = torch.diag_embed(cutoffs)

            A += C

            # now set up the linear system
            A = torch.cat(
                (A, torch.ones((n_mol_edges, 1), device=positions.device)), dim=1
            )
            A = torch.cat(
                (A, torch.ones((1, n_mol_edges + 1), device=positions.device)), dim=0
            )

            b = -1 * T_mol.t() @ eneg[molecule_indices]
            b = b.squeeze()
            b = torch.cat(
                (b, torch.tensor([total_charge], device=positions.device)), dim=0
            )

            # let's try a basic linear solve
            x = torch.linalg.solve(A, b)
            x = x[:-1].squeeze()
            x = T_mol @ x
            output_partial_charges.append(x)
            # print("pratial charges", x)

            # check sum of charges is equal to total charge
            # print("Sum of partial charges")
            # print(torch.sum(x), total_charge)

        return torch.cat(output_partial_charges, dim=0)
