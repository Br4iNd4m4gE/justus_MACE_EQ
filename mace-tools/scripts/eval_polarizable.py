###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import ast

import ase.data
import ase.io
import numpy as np
import torch

from macetools import data
from mace.tools import torch_geometric, torch_tools, utils

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output", help="output path", required=True)
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
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dont_compute_force",
        help="compute stress",
        action="store_true",
    )
    parser.add_argument(
        "--return_contributions",
        help="model outputs energy contributions for each body order, only suppported for MACE, not ScaleShiftMACE",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    parser.add_argument(
        "--scf_history",
        help="how much of the scf history to return",
        type=str,
        default="absolute_change",
        choices=[
            "none",
            "absolute_change",
            "full_history"
        ]
    )
    parser.add_argument(
        "--multipole_l",
        help="l",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--external_field_key",
        help="key for external field",
        type=str,
        default="external_field",
    )
    parser.add_argument(
        "--fermi_level_key",
        help="key for external field",
        type=str,
        default="fermi_level",
    )
    parser.add_argument(
        "--scf_training_options", type=str, default=None
    )
    parser.add_argument(
        "--use_pbc_evaluator",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # parse special options for scf model training
    if args.scf_training_options is not None:
        scf_training_options = ast.literal_eval(args.scf_training_options)
        if not "mixing_parameter" in scf_training_options.keys():
            scf_training_options["mixing_parameter"] = 1.0
    else:
        scf_training_options = None

    compute_force = not args.dont_compute_force

    # Load model
    model = torch.load(f=args.model, map_location=args.device)
    model = model.to(
        args.device
    )  # shouldn't be necessary but seems to help wtih CUDA problems

    for param in model.parameters():
        param.requires_grad = False

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, index=":")
    configs = [data.config_from_atoms(
        atoms, 
        density_coefficients_max_l=args.multipole_l,
        density_coefficients_key="AIMS_atom_multipoles",
        external_field_key=args.external_field_key,
        fermi_level_key=args.fermi_level_key,
    ) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max)
            )
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    contributions_list = []
    stresses_list = []
    forces_collection = []
    density_coefficients_collection = []
    dipoles_list = []
    polarizations_list = []
    full_charge_histories_list = []
    electrostatic_energies_list = []
    electron_energies_list = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(
            batch.to_dict(), 
            compute_stress=args.compute_stress, 
            compute_force=compute_force, 
            num_scf_steps=scf_training_options["num_scf_steps"], 
            constant_charge=scf_training_options["constant_charge"],
            mixing_parameter=scf_training_options["mixing_parameter"],
            use_pbc_evaluator=args.use_pbc_evaluator,
        )

        energies_list.append(torch_tools.to_numpy(output["energy"]))
        if args.compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))

        if args.return_contributions:
            contributions_list.append(torch_tools.to_numpy(output["contributions"]))

        if compute_force:
            forces = np.split(
                torch_tools.to_numpy(output["forces"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )

            forces_collection.append(forces[:-1])  # drop last as its emtpy

        # output["charges_history"] is [n_atoms, n_components, n_scf]
        charges_history = np.split(
            torch_tools.to_numpy(output["charges_history"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        # this will just be a list of arrays one for each atom
        full_charge_histories_list += charges_history[:-1]

        # for now, all the models in this repo return the density coefficients
        density_coefficients = np.split(
            torch_tools.to_numpy(output["density_coefficients"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        density_coefficients_collection.append(density_coefficients[:-1])
        
        dipoles_list.append(torch_tools.to_numpy(output["dipole"]))

        if "electrostatic_energy" in output.keys():
            electrostatic_energies_list.append(torch_tools.to_numpy(output["electrostatic_energy"]))
        if "electron_energy" in output.keys():
            electron_energies_list.append(torch_tools.to_numpy(output["electron_energy"]))

    energies = np.concatenate(energies_list, axis=0)
    dipoles = np.concatenate(dipoles_list, axis=0)
    if "polarization" in output.keys():
        polarizations = np.concatenate(polarizations_list, axis=0)
    if "electrostatic_energy" in output.keys():
        electrostatic_energies = np.concatenate(electrostatic_energies_list, axis=0)
    if "electron_energy" in output.keys():
        electron_energies = np.concatenate(electron_energies_list, axis=0)
    
    forces_list = [
        forces for forces_list in forces_collection for forces in forces_list
    ]
    density_coefficients_list = [
        density_coefficients
        for density_coefficients_list in density_coefficients_collection
        for density_coefficients in density_coefficients_list
    ]
    assert len(atoms_list) == len(energies) == len(density_coefficients_list) == dipoles.shape[0]

    if args.compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(atoms_list) == stresses.shape[0]

    if args.return_contributions:
        contributions = np.concatenate(contributions_list, axis=0)
        assert len(atoms_list) == contributions.shape[0]

    if args.scf_history != 'none':
        assert len(atoms_list) == len(full_charge_histories_list)
        # each item of full_charge_histories_list is [n_atoms, n_components, n_scf]
        # calculate the average different with repsect to the final step
        average_delta = [
            [
                np.average(np.abs(
                    single_config_data[...,i] - single_config_data[...,-1]
                ))
                for i in range(single_config_data.shape[-1])
            ]
            for single_config_data in full_charge_histories_list
        ]
        scf_convergence_relative = average_delta
        
        if args.scf_history == 'absolute_change' or args.scf_history == 'full_history':
            reshaped_charge_histories = [
                np.hstack([subarr for subarr in arr]) for arr in full_charge_histories_list
            ]
            scf_convergence_approach = []
            for single_ats_charge_history in full_charge_histories_list:
                abs_average_charge_change = [
                    np.average(np.abs(single_ats_charge_history[...,0]))
                ]
                for i in range(single_ats_charge_history.shape[-1] - 1):
                    abs_average_charge_change.append(
                        np.average(np.abs(
                            single_ats_charge_history[...,i] - single_ats_charge_history[...,i+1]
                        ))
                    )
                scf_convergence_approach.append(abs_average_charge_change)
        if args.scf_history == 'full_history':
            reshaped_charges = [
                np.asarray([np.reshape(single_atoms_charges, (-1), order='F') for single_atoms_charges in config_charges]) for config_charges in full_charge_histories_list
            ]
            print(type(reshaped_charges))

    # Store data in atoms objects
    for i, (atoms, energy, density_coefficients, dipole) in enumerate(
        zip(atoms_list, energies, density_coefficients_list, dipoles)
    ):
        atoms.calc = None  # crucial
        atoms.info[args.info_prefix + "energy"] = energy
        if compute_force:
            forces = forces_list[i]
            atoms.arrays[args.info_prefix + "forces"] = forces
        atoms.arrays[args.info_prefix + "density_coefficients"] = density_coefficients
        atoms.info[args.info_prefix + "dipole"] = dipole

        if args.compute_stress:
            atoms.info[args.info_prefix + "stress"] = stresses[i]

        if args.return_contributions:
            atoms.info[args.info_prefix + "BO_contributions"] = contributions[i]

        if args.scf_history == 'absolute_change':
            atoms.info['scf_convergence'] = np.asarray(scf_convergence_approach[i])
            atoms.info['scf_convergence_relative'] = np.asarray(scf_convergence_relative[i])
        elif args.scf_history == 'full_history':
            atoms.arrays['scf_charge_history'] = reshaped_charges[i]
            atoms.info['scf_convergence'] = np.asarray(scf_convergence_approach[i])
            atoms.info['scf_convergence_relative'] = np.asarray(scf_convergence_relative[i])

        if "electrostatic_energy" in output.keys():
            atoms.info[args.info_prefix + "electrostatic_energy"] = electrostatic_energies[i]
        if "electron_energy" in output.keys():
            atoms.info[args.info_prefix + "electron_energy"] = electron_energies[i]


    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
