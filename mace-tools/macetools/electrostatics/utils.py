import torch
from typing import Dict, List , Optional
from mace.tools.scatter import scatter_sum
from scipy.constants import c, e
import scipy
import numpy as np


def compute_fixed_charge_dipole(
    charges: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    mu = positions * charges.unsqueeze(-1) / (1e-11 / c / e)  # [N_atoms,3]
    return scatter_sum(
        src=mu, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs
    )  # [N_graphs,3]


def compute_total_charge_dipole(
    density_coefficients: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
):
    dipole_contribution = positions * density_coefficients[:,:1]

    dipole = scatter_sum(
        src=dipole_contribution, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs
    )

    if density_coefficients.shape[1] > 1:
        dipole_p = scatter_sum(
            src=density_coefficients[...,1:4], index=batch, dim=-2, dim_size=num_graphs
        )
        dipole = dipole + torch.gather(dipole_p, -1, torch.tensor([[2,0,1]])) # CS phase convention

    total_charge = scatter_sum(
        src=density_coefficients[:,0], index=batch, dim=-1#, dim_size=num_graphs
    )

    return total_charge, dipole


def compute_polarization(
    density_coefficients: torch.Tensor,
    edge_fluxes: torch.Tensor,
    edge_vectors: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
):
    # flux piece
    edge_dipoles = edge_fluxes.unsqueeze(-1) * edge_vectors
    sender, receiver = edge_index
    total_flux = scatter_sum(
        src=edge_dipoles, index=batch[sender], dim=-2, dim_size=num_graphs
    )

    #print("charges piece:", total_flux)

    # dipole piece
    if density_coefficients.shape[1] > 1:
        dipole_p = scatter_sum(
            src=density_coefficients[...,1:4], index=batch, dim=-2, dim_size=num_graphs
        )
        #print("dipoles piece:", dipole_p[...,[2,0,1]])
        total_flux = total_flux + dipole_p[...,[2,0,1]] # CS phase convention
    #print("added: ", total_flux)

    return total_flux


def compute_coulomb_energy(
    partial_charges: torch.Tensor, data: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Compute the coulomb energy of a system of partial charges"""
    # compute the pairwise distances
    # compute the distances, accounting for pbc
    posn = data["positions"]
    batch_indices = data["batch"]

    output_energies = []
    k_e = 14.399645478425668

    for idx in torch.unique(batch_indices):
        # get the positions of the atoms in the current molecule
        molecule_mask = batch_indices == idx
        positions = posn[molecule_mask]
        molecule_partial_charges = partial_charges[molecule_mask]
        # iterate over each molecule in the batch

        # are the distance accounting for pbc? No

        distances = torch.cdist(positions, positions)
        # put ones on the diagonal to avoid dividing by zero
        distances = distances + torch.eye(distances.shape[0], device=distances.device)

        # change all distances greater than the cutoff to infinity, use a 1 angstrom cutoff
        # compute the coulomb energy
        potential = (
            k_e
            * torch.outer(molecule_partial_charges, molecule_partial_charges)
            * (torch.erf(distances / 1) / distances)
        )

        potential = torch.triu(potential, diagonal=1)
        # sum the values to get the total energy
        potential_energy = torch.sum(potential)
        # print("potential energy", potential_energy)
        # print(potential)
        # print(coulomb_energy)
        # print("final potential", coulomb_energy)
        output_energies.append(potential_energy)

    output_energies = torch.stack(output_energies)  # [n_graphs])
    return output_energies

def compute_qmmm_forces(
    energy: torch.Tensor,
    positions: torch.Tensor,
    esp: torch.Tensor,
    esp_grad: torch.Tensor,
    compute_force: bool = True,
) -> torch.Tensor:
    """Compute the QM/MM forces from the ESP and the energy"""
    if not compute_force:
        return torch.zeros_like(positions)
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    dE_dESP = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[esp],  # [n_nodes, 1]
        grad_outputs=grad_outputs,
        retain_graph=True,  # Make sure the graph is not destroyed for actual force computation
        allow_unused=False,
    )[0]  # [n_nodes, 1]
    if dE_dESP is None:
        return torch.zeros_like(esp_grad)
    dE_dr = dE_dESP * esp_grad  # [n_nodes, 3]
    return -1 * dE_dr