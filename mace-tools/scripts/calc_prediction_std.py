#!/usr/bin/env python
import argparse
from datetime import timedelta
import os
import time
from typing import Sequence
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "graph_longrange"))

from mace.tools import torch_tools, set_default_dtype
from macetools import data
from macetools.utils.extend_arg_parse import extended_arg_parser
from macetools.utils.constants import atomic_number_to_element


from load_data import load_model, load_dataset

def parse_args():
    ap = extended_arg_parser()
    ap.add_argument(
        "-p",
        "--prefix",
        help="Prefix of the model to load",
        type=str,
        default="model_",
    )
    args = ap.parse_args()
    return args

def load_models(args: argparse.Namespace) -> Sequence[torch.nn.Module]:
    prefix = args.prefix
    prefix_dirname = os.path.dirname(prefix)
    prefix_basename = os.path.basename(prefix)
    if prefix_dirname == "":
        prefix_dirname = os.getcwd()
    
    model_paths = [file for file in os.listdir(prefix_dirname) if file.startswith(prefix)]
    model_paths.sort()
    
    if len(model_paths) == 0:
        raise FileNotFoundError(f"No models found with prefix {prefix_basename} in {prefix_dirname}")
    print(f"Found {len(model_paths)} models with prefix {prefix_basename} in {prefix_dirname}")
    print(f"Loading models from {model_paths}")
    
    models = []
    for model_path in model_paths:
        model = load_model(model_path, args.device)
        models.append(model)

    r_max = models[0].r_max.item()
    atomic_numbers = torch_tools.to_numpy(models[0].atomic_numbers)
    atomic_energies = torch_tools.to_numpy(models[0].atomic_energies_fn.atomic_energies)

    for model in models:
        assert model.r_max.item() == r_max, f"r_max is not the same for all models: {model.r_max.item()} != {r_max}"
        assert np.all(torch_tools.to_numpy(model.atomic_numbers) == atomic_numbers), f"atomic_numbers are not the same for all models, {model.atomic_numbers} != {atomic_numbers}"
        assert np.all(torch_tools.to_numpy(model.atomic_energies_fn.atomic_energies) == atomic_energies), f"atomic_energies are not the same for all models: {model.atomic_energies_fn.atomic_energies} != {atomic_energies}"

    args.r_max = r_max
    args.E0s = repr({atomic_number: atomic_energy for atomic_number, atomic_energy in zip(atomic_numbers, atomic_energies)})
    return models

def main():
    args = parse_args()
    set_default_dtype(args.default_dtype)
    np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x}, suppress=True)
    models = load_models(args)
    print("Models loaded successfully")

    collections, _, _, test_loader, z_table, _ = load_dataset(args)
    print("Test dataset loaded successfully")

    output_args = {
            "energy": True,
            "forces": True,
            "virials": False,
            "stress": False,
    }
    keys = ["charges", "energy", "forces"]

    references = {key: [] for key in keys}
    test_outputs_models = [] # list of outputs for each batch for each model
    for test_batch in test_loader:
        test_batch = test_batch.to(args.device)
        test_batch_dict = test_batch.to_dict()
        for key in keys:
            references[key].append(test_batch_dict[key])
        output_batch = []
        for model in models:
            output_model = model(
                        test_batch_dict,
                        training=False,
                        compute_force=output_args["forces"],
                        compute_virials=output_args["virials"],
                        compute_stress=output_args["stress"],
                        use_pbc_evaluator=False
                    )
            output_batch.append(output_model)
        test_outputs_models.append(output_batch)

    # Concatenate the references for the different batches
    for key in keys:
        references[key] = torch.cat(references[key], dim=0)

    # Stack the results for the different models for predicted charges, energies and forces
    key_outputs = []
    for key in keys:
        output_batches = []
        for output_batch in test_outputs_models:
            output_models = []
            for output_model in output_batch:
                numpy_output = torch_tools.to_numpy(output_model[key])
                output_models.append(numpy_output)
            stacked_output_models = np.stack(output_models, axis=0)
            output_batches.append(stacked_output_models)
        concatenated_output_batches = np.concatenate(output_batches, axis=1)
        key_outputs.append(concatenated_output_batches)

    ref_charges, ref_energies, ref_forces = references["charges"], references["energy"], references["forces"] # shape(n_molecules,n_atoms), shape(n_molecules,), shape(n_molecules,n_atoms,3)
    predicted_charges, predicted_energies, predicted_forces = key_outputs # shape(n_models,n_molecules,n_atoms), shape(n_models,n_molecules,), shape(n_models,n_molecules,n_atoms,3)

    ref_charges = torch_tools.to_numpy(ref_charges).flatten() # shape(n_molecules*n_atoms)
    ref_energies = torch_tools.to_numpy(ref_energies).flatten() # shape(n_molecules)
    ref_forces = torch_tools.to_numpy(ref_forces).flatten() # shape(n_molecules*n_atoms*3)
    predicted_charges = predicted_charges.reshape(predicted_charges.shape[0], -1)
    predicted_energies = predicted_energies.reshape(predicted_energies.shape[0], -1)
    predicted_forces = predicted_forces.reshape(predicted_forces.shape[0], -1)

    charge_rmse = np.sqrt(np.mean((predicted_charges - ref_charges)**2, axis=1)) # shape(n_models)
    energy_rmse = np.sqrt(np.mean((predicted_energies - ref_energies)**2, axis=1)) # shape(n_models)
    force_rmse = np.sqrt(np.mean((predicted_forces - ref_forces)**2, axis=1)) # shape(n_models)

    charge_mean = np.mean(predicted_charges, axis=0) # shape(n_molecules)
    energy_mean = np.mean(predicted_energies, axis=0) # shape(n_molecules)
    force_mean = np.mean(predicted_forces, axis=0) # shape(n_molecules)

    charge_std = np.std(predicted_charges, axis=0) # shape(n_molecules)
    energy_std = np.std(predicted_energies, axis=0) # shape(n_molecules)
    force_std = np.std(predicted_forces, axis=0) # shape(n_molecules)

    mean_charge_mean = np.mean(charge_mean) # shape(1)
    mean_energy_mean = np.mean(energy_mean) # shape(1)
    mean_force_mean = np.mean(force_mean) # shape(1)

    mean_charge_std = np.mean(charge_std) # shape(1)
    mean_energy_std = np.mean(energy_std) # shape(1)
    mean_force_std = np.mean(force_std) # shape(1)

    std_charge_std = np.std(charge_std) # shape(1)
    std_energy_std = np.std(energy_std) # shape(1)
    std_force_std = np.std(force_std) # shape(1)

    max_charge_std = np.max(charge_std) # shape(1)
    max_energy_std = np.max(energy_std) # shape(1)
    max_force_std = np.max(force_std) # shape(1)

    energy_threshold = mean_energy_std + 5*std_energy_std
    force_threshold = mean_force_std + 10*std_force_std

    print(f"Shape of reference charges: {ref_charges.shape}")
    print(f"Shape of reference energies: {ref_energies.shape}")
    print(f"Shape of reference forces: {ref_forces.shape}")

    print(f"Shape of predicted charges: {predicted_charges.shape}")
    print(f"Shape of predicted energies: {predicted_energies.shape}")
    print(f"Shape of predicted forces: {predicted_forces.shape}")

    print(f"Mean Charge RMSE: {charge_rmse}")
    print(f"Mean Energy RMSE: {energy_rmse}")
    print(f"Mean Force RMSE: {force_rmse}")

    print(f"Mean Charge Mean: {mean_charge_mean: .4f}")
    print(f"Mean Energy Mean: {mean_energy_mean: .4f}")
    print(f"Mean Force Mean: {mean_force_mean: .4f}")

    print(f"Mean Charge Std: {mean_charge_std: .4f}")
    print(f"Mean Energy Std: {mean_energy_std: .4f}")
    print(f"Mean Force Std: {mean_force_std: .4f}")

    print(f"Std Charge Std: {std_charge_std: .4f}")
    print(f"Std Energy Std: {std_energy_std: .4f}")
    print(f"Std Force Std: {std_force_std: .4f}")

    print(f"Max Charge Std: {max_charge_std: .4f}")
    print(f"Max Energy Std: {max_energy_std: .4f}")
    print(f"Max Force Std: {max_force_std: .4f}")

    print(f"Suggested Energy Threshold: {energy_threshold: .4f}")
    print(f"Suggested Force Threshold: {force_threshold: .4f}")

    energy_dict = {
        "Energy Std": np.reshape(energy_std,(-1,))
    }
    energy_df = pd.DataFrame(energy_dict)
    sns.histplot(data=energy_df, x="Energy Std")
    plt.xlabel('Energy Standard Deviation')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("energy_std_histogram.png")
    plt.close()

    present_atomic_numbers = torch_tools.to_numpy(models[0].atomic_numbers)
    one_hot_encoded_atom_numbers = np.concatenate([torch_tools.to_numpy(batch.to_dict()["node_attrs"]) for batch in test_loader], axis=0)
    atomic_numbers = present_atomic_numbers[np.argmax(one_hot_encoded_atom_numbers, axis=1)]

    force_dict = {
        "Force Std": np.reshape(force_std,(-1,)),
        "Atom Types": np.array(atomic_numbers).repeat(3).flatten()
    }

    force_df = pd.DataFrame(force_dict)
    force_df["Atom Types"] = force_df["Atom Types"].replace(atomic_number_to_element)

    sns.histplot(data=force_df, x="Force Std", hue="Atom Types", stat='probability', common_norm=False)
    plt.xlabel('Force Standard Deviation')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("force_std_histogram.png")
    plt.close()


if __name__ == "__main__":
    main()