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

from lukas_train import initialize_model


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
        action="store_true",
        default=False,
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
        "--multipole_l",
        help="l",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use_pbc_evaluator",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device).eval()
    model = model.to(
        args.device
    )  # shouldn't be necessary but seems to help wtih CUDA problems

    for param in model.parameters():
        param.requires_grad = False

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, index=":")
    configs = [data.config_from_atoms( # Property keys are not passed properly, everything non-default is zeros
        atoms, 
        density_coefficients_max_l=args.multipole_l,
        density_coefficients_key="AIMS_atom_multipoles"
    ) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    # atomic_energies = model.atomic_energies_fn.atomic_energies

    # old_args = argparse.Namespace(model='maceQEq', r_max=8.0, radial_type='bessel', num_radial_basis=8, num_cutoff_basis=5, pair_repulsion=False, distance_transform='None', interaction='RealAgnosticResidualInteractionBlock', interaction_first='RealAgnosticResidualInteractionBlock', max_ell=3, correlation=3, num_interactions=2, MLP_irreps='16x0e', radial_MLP='[64, 64, 64]', hidden_irreps='64x0e+64x1o', num_channels=None, max_L=None, gate='silu', scaling='rms_forces_scaling', avg_num_neighbors=1, compute_avg_num_neighbors=True, compute_stress=False, compute_forces=True, multi_processed_test=False, num_workers=0, pin_memory=True, atomic_numbers=None, mean=None, std=None, statistics_file=None, E0s='{1: -13.575035506869515, 6: -1029.6173622986487, 7: -1485.1410643783852, 8: -2042.617308911902, 16: -10832.265333248919}', keep_isolated_atoms=False, huber_delta=0.01, optimizer='adam', batch_size=10, valid_batch_size=10, lr=0.01, swa_lr=0.001, weight_decay=5e-07, amsgrad=True, scheduler='ReduceLROnPlateau', lr_factor=0.8, scheduler_patience=50, lr_scheduler_gamma=0.9993, swa=True, start_swa=None, ema=True, ema_decay=0.99, max_num_epochs=100, patience=2048, foundation_model=None, foundation_model_readout=True, eval_interval=1, keep_checkpoints=False, save_all_checkpoints=False, restart_latest=True, save_cpu=True, clip_grad=10.0, wandb=True, scale_atsize=1.0, dont_add_coulomb=False, include_local_electron_energy=False, atomic_multipoles_max_l=0, atomic_multipoles_smearing_width=1.0, kspace_cutoff_factor=1.5, atomic_formal_charges=None, formal_charges_from_data=True, field_feature_widths=[1.0], dont_include_pbc_corrections=False, include_electrostatic_self_interaction=False, valid_set_seed=None, scf_training_options=None, field_dependence_type='local_linear', final_field_readout_type='StrictQuadraticFieldEnergyReadout', quadrupole_feature_corrections=False, use_pbc_evaluator=False)
    # old_model = initialize_model(args=old_args, z_table=z_table, atomic_energies=atomic_energies)
    # old_model.load_state_dict(model.state_dict())
    # old_model.eval()
    # old_model.to(device)
    # model = old_model

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
    ref_energies_list = []
    pred_energies_list = []
    contributions_list = []
    stresses_list = []
    ref_forces_list = []
    pred_forces_collection = []
    charges_collection = []
    dipoles_list = []
    polarizations_list = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(
            batch.to_dict(), compute_stress=args.compute_stress, compute_force=not args.dont_compute_force, use_pbc_evaluator=args.use_pbc_evaluator
        )
        batch = batch.to("cpu")
        ref_energies_list.append(batch.energy.numpy())
        pred_energies_list.append(torch_tools.to_numpy(output["energy"]))
        if args.compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))

        if args.return_contributions:
            contributions_list.append(torch_tools.to_numpy(output["contributions"]))

        ref_forces = np.split(
            torch_tools.to_numpy(batch.forces),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        ref_forces_list.append(ref_forces[:-1])  # drop last as its empty
        pred_forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        pred_forces_collection.append(pred_forces[:-1])  # drop last as its emtpy

        # for now, all the models in this repo return the density coefficients
        pred_charges = np.split(
            torch_tools.to_numpy(output["charges"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        charges_collection.append(pred_charges[:-1])
        
        dipoles_list.append(torch_tools.to_numpy(output["dipole"]))

    ref_energies = np.concatenate(ref_energies_list, axis=0)
    pred_energies = np.concatenate(pred_energies_list, axis=0)
    pred_dipoles = np.concatenate(dipoles_list, axis=0)
    ref_forces_list = [
        forces for forces_list in ref_forces_list for forces in forces_list
    ]
    pred_forces_list = [
        forces for forces_list in pred_forces_collection for forces in forces_list
    ]
    
    # Add this line to create pred_charges_list
    pred_charges_list = [
        charges for charges_list in charges_collection for charges in charges_list
    ]
    
    # Fix the qmmm_forces_list initialization
    qmmm_forces_collection = []  # This appears to be empty in your case
    if len(qmmm_forces_collection) == 0:
        # Create a list of zero arrays with the same shapes as pred_forces_list
        qmmm_forces_list = [np.zeros_like(forces) for forces in pred_forces_list]
    else:
        qmmm_forces_list = [
            forces for forces_list in qmmm_forces_collection for forces in forces_list
        ]

    assert len(atoms_list) == len(ref_energies) == len(ref_forces_list)
    assert len(atoms_list) == len(pred_energies) == len(pred_forces_list) == len(pred_charges_list) == len(pred_dipoles)

    if args.compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(atoms_list) == stresses.shape[0]

    if args.return_contributions:
        contributions = np.concatenate(contributions_list, axis=0)
        assert len(atoms_list) == contributions.shape[0]

    # Calculate and print errors
    rmse_energy = np.sqrt(np.mean((ref_energies - pred_energies) ** 2))
    mae_energy = np.mean(np.abs(ref_energies - pred_energies))
    np.set_printoptions(precision=4)
    print(f"Min Max reference energy: {np.min(ref_energies)} eV, {np.max(ref_energies)} eV")
    print(f"Min Max predicted energy: {np.min(pred_energies)} eV, {np.max(pred_energies)} eV")
    print(f"RMSE energy: {rmse_energy} eV")
    print(f"MAE energy: {mae_energy} eV")

    ref_flat_forces = np.concatenate(ref_forces_list, axis=0).flatten()
    pred_flat_forces = np.concatenate(pred_forces_list, axis=0).flatten()
    rmse_force = np.sqrt(
        np.mean((ref_flat_forces - pred_flat_forces) ** 2, axis=0)
    )
    mae_force = np.mean(np.abs(ref_flat_forces - pred_flat_forces), axis=0)
    print(f"Min Max reference forces: {np.min(ref_flat_forces)} eV/Å, {np.max(ref_flat_forces)} eV/Å")
    print(f"Min Max predicted forces: {np.min(pred_flat_forces)} eV/Å, {np.max(pred_flat_forces)} eV/Å")
    print(f"RMSE force: {rmse_force} eV/Å")
    print(f"MAE force: {mae_force} eV/Å")

    # Store data in atoms objects
    for i, (atoms, pred_energy, pred_forces, pred_charges
    , pred_dipole) in enumerate(
        zip(atoms_list, pred_energies, pred_forces_list, pred_charges_list, pred_dipoles)
    ):
        atoms.calc = None  # crucial
        atoms.info[args.info_prefix + "energy"] = pred_energy
        atoms.arrays[args.info_prefix + "forces"] = pred_forces
        atoms.arrays[args.info_prefix + "charges"] = pred_charges
        atoms.info[args.info_prefix + "dipole"] = pred_dipole

        if args.compute_stress:
            atoms.info[args.info_prefix + "stress"] = stresses[i]

        if args.return_contributions:
            atoms.info[args.info_prefix + "BO_contributions"] = contributions[i]

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
