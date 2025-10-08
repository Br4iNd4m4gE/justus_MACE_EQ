###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse

import ase.data
import ase.io
import numpy as np
import torch

from macetools import data
from mace.tools import torch_geometric, torch_tools, utils

from graph_longrange.gto_electrostatics import (
    PBCAgnosticDirectElectrostaticEnergyBlock,
    gto_basis_kspace_cutoff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--atomic_multipoles_smearing_width",
        help="prefix for energy, forces and stress keys",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--kspace_cutoff_factor",
        help="prefix for energy, forces and stress keys",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--atomic_multipoles_max_l",
        help="l",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--atomic_multipoles_key",
        type=str,
        default="AIMS_atom_multipoles",
    )
    parser.add_argument(
        "--info_prefix",
        type=str,
        default="electrostatic_",
    )
    parser.add_argument(
        "--include_self_interaction",
        action="store_true",
    )
    parser.add_argument(
        "--remove_corrections",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, index=":")
    configs = [data.config_from_atoms(
        atoms, 
        density_coefficients_max_l=args.atomic_multipoles_max_l,
        density_coefficients_key=args.atomic_multipoles_key
    ) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in range(83)])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(5.0)
            )
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []

    kspace_cutoff = args.kspace_cutoff_factor * gto_basis_kspace_cutoff(
        [args.atomic_multipoles_smearing_width], args.atomic_multipoles_max_l
    )

    coulomb_energy = PBCAgnosticDirectElectrostaticEnergyBlock(
        density_max_l=args.atomic_multipoles_max_l,
        density_smearing_width=args.atomic_multipoles_smearing_width,
        kspace_cutoff=kspace_cutoff,
        include_pbc_corrections=not(args.remove_corrections),
        include_self_interaction=args.include_self_interaction
    )
    coulomb_energy.to(device)

    for batch in data_loader:
        batch = batch.to(device)
        batch =  batch.to_dict()
        num_graphs = batch["ptr"].numel() - 1

        qs = batch['density_coefficients']
        #qs[:,1:] = qs[:,[1,2,0]]

        electro_energy = coulomb_energy(
            qs,
            batch["positions"],
            batch["batch"],
            batch["cell"].view(-1, 3, 3),
            batch["rcell"].view(-1, 3, 3),
            batch["volume"],
            batch["pbc"].view(-1, 3),
            num_graphs,
        )

        energies_list.append(torch_tools.to_numpy(electro_energy))

    energies = np.concatenate(energies_list, axis=0)

    # Store data in atoms objects
    for i, (atoms, energy) in enumerate(zip(atoms_list, energies)):
        atoms.calc = None  # crucial
        atoms.info[args.info_prefix + "energy"] = energy

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
