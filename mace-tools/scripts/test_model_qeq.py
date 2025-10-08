import argparse
import dataclasses
from datetime import timedelta
import logging
import os
import time
import warnings
from scipy.spatial import distance_matrix
from sklearn.preprocessing import OneHotEncoder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import torch

import mace
from mace import tools
from mace.tools import torch_geometric, torch_tools

# macetools replaces data, and some utils
from macetools import data
import macetools.utils

from load_data import load_dataset

#MODEL_PATHS = ["../../../model_energy_force_mace0.pt"]
MODEL_PATHS = ["model_energy_force0.pt","model_energy_force1.pt","model_energy_force2.pt"]

STARTING_STRUCTURE_IDX = 0 # Index of the starting structure for the simulation run in the dataset
TEST_IDX = 0 # Dummy index for the test dataset, always 0

def main():
    # Setup
    parser = macetools.utils.extended_arg_parser()
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    args = parser.parse_args()
    tag = tools.get_tag(name=args.name, seed=args.seed)
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.info(f"Configuration: {args}")
    np.set_printoptions(precision=4, suppress=True)

    # Model preparation
    models = [torch.jit.load(model_path) for model_path in MODEL_PATHS]
    args.r_max = models[0].r_max.item()
    atomic_numbers = models[0].atomic_numbers.numpy()
    atomic_energies = models[0].atomic_energies_fn.atomic_energies.numpy()
    args.E0s = repr({atomic_number: atomic_energy for atomic_number, atomic_energy in zip(atomic_numbers, atomic_energies)})
    
    logging.info(f"Model r_max: {args.r_max}")
    logging.info(f"Model E0s: {args.E0s}")

    # Data preparation
    args.batch_size = 1 # Only one batch is needed
    args.vaid_batch_size = 1
    original_collections, _, valid_loader, _, _, _ = load_dataset(args) # Dont't use the test_loader, because it is shuffled

    args.train_file = args.test_file
    args.valid_file = args.test_file
    args.energy_key = args.info_prefix + "energy"
    args.forces_key = args.info_prefix + "force"
    args.stress_key = args.info_prefix + "stress"
    args.virials_key = args.info_prefix + "virials"
    args.dipole_key = args.info_prefix + "dipole"
    args.charges_key = args.info_prefix + "charge"

    predicted_collections, _, _, test_loader, _, _ = load_dataset(args)


    # Data preparation
    one_hot_encoded_elements = np.loadtxt("atomic_numbers.txt", dtype=np.int64)
    written_bond_indices = np.loadtxt("edge_indices.txt", dtype=np.int64).T
    written_shifts = np.loadtxt("shifts.txt", dtype=np.float64)
    written_unit_shifts = np.loadtxt("unit_shifts.txt", dtype=np.float64)

    n_atoms = len(one_hot_encoded_elements) 
    # written_charges = np.genfromtxt("output.txt", max_rows=n_atoms, skip_header=0, dtype=np.float64)
    # written_energy = np.genfromtxt("output.txt", skip_header=n_atoms, max_rows=1, dtype=np.float64)
    # written_forces = np.genfromtxt("output.txt", skip_header=n_atoms+1, max_rows=n_atoms, dtype=np.float64)

    output_args = {
            "energy": True,
            "forces": True,
            "virials": False,
            "stress": False,
    }

    for batch_idx, train_batch in enumerate(valid_loader):
        if batch_idx != STARTING_STRUCTURE_IDX: # valid_loader is ordered as the dataset
            continue

        train_batch_dict = train_batch.to_dict()
        break # Only one batch is needed

    test_outputs_models = []
    for batch_idx, test_batch in enumerate(test_loader):
        if batch_idx != TEST_IDX:
            continue
        test_batch = test_batch.to(device)
        test_batch_dict = test_batch.to_dict()
        for model in models:
            output_model = model(
                        test_batch_dict,
                        training=False,
                        compute_force=output_args["forces"],
                        compute_virials=output_args["virials"],
                        compute_stress=output_args["stress"],
                        use_pbc_evaluator=False
                    )
            test_outputs_models.append(output_model)
        break # Only one batch is needed

    # Stack the results for the different models for predicted charges, energies and forces
    keys = ["charges", "energy", "forces"]
    predicted_charges, predicted_energies, predicted_forces = [
        np.stack([torch_tools.to_numpy(output[key]) for output in test_outputs_models], axis=0)
        for key in keys
    ] # shapes(n_models,n_atoms), (n_models, 1), (n_models,n_atoms,3)
    written_charges, written_energy, written_force = [
        torch_tools.to_numpy(test_batch_dict[key]) for key in keys
    ]
    print("Predicted shapes:", predicted_charges.shape, predicted_energies.shape, predicted_forces.shape)
    print("Written Shapes:", written_charges.shape, written_energy.shape, written_force.shape)

    # Assert the input and output of the model
    reference_charges = train_batch_dict["charges"].numpy()
    reference_energy = train_batch_dict["energy"].numpy()
    reference_forces = train_batch_dict["forces"].numpy()

    run_input_assertions(train_batch_dict, test_batch_dict, one_hot_encoded_elements, written_bond_indices, written_shifts, written_unit_shifts)
    run_output_assertions(predicted_charges, predicted_energies, predicted_forces, written_charges, written_energy, written_force, reference_charges, reference_energy, reference_forces)

    if len(models) == 1:
        print("Only one model found. Skipping std assertions")
    elif not os.path.exists("qm_mlmm_std.xyz"):
        print("qm_mlmm_std.xyz not found. Skipping std assertions")
    else:
        run_std_assertions(predicted_energies, predicted_forces, n_atoms)

def run_input_assertions(train_batch_dict, test_batch_dict, one_hot_encoded_elements, written_bond_indices, written_shifts, written_unit_shifts):
    print("-----------------Input assertion--------------------------")
    starting_geom = train_batch_dict["positions"].detach().numpy()
    written_geom = test_batch_dict["positions"].detach().numpy()
    n_atoms = len(starting_geom)
    starting_distances = distance_matrix(starting_geom, starting_geom)
    input_distances = distance_matrix(written_geom, written_geom)
    inputs = {
        "node_attrs": [train_batch_dict["node_attrs"].detach().numpy().astype(np.int32), one_hot_encoded_elements],
        "positions": [input_distances, starting_distances],
        "edge_index": [train_batch_dict["edge_index"].detach().numpy(), written_bond_indices],
        "shifts": [train_batch_dict["shifts"].detach().numpy(), written_shifts],
        "unit_shifts": [train_batch_dict["unit_shifts"].detach().numpy(), written_unit_shifts],
        "cell": [train_batch_dict["cell"].detach().numpy(), test_batch_dict["cell"].detach().numpy()],
        "esp": [train_batch_dict["esp"].detach().numpy(), test_batch_dict["esp"].detach().numpy()],
        "esp_gradient": [train_batch_dict["esp_gradient"].detach().numpy(), test_batch_dict["esp_gradient"].detach().numpy()],
    }
    for key, value in inputs.items():
        try:
            if key == "edge_index":
                assert value[0].shape[0] == value[1].shape[0], f"{key} Assertion possibly failed because of shape mismatch"
                assert np.allclose(value[0], value[1], atol=1e-03), f"{key} Assertion possibly failed because of value mismatch"
            elif key == "esp" or key == "esp_gradient":
                assert np.allclose(value[0], value[1], atol=1e-02), f"{key} Assertion failed"
            else:
                assert np.allclose(value[0], value[1], atol=1e-03), f"{key} Assertion failed"
        except AssertionError as error:
            print(key)
            print(error)
            if key == "positions":
                with np.printoptions(precision=2):
                    min_n_atoms = min(5, n_atoms)
                    print(f"Geoms for first {min_n_atoms} atoms")
                    print(starting_geom[0:min_n_atoms])
                    print(written_geom[0:min_n_atoms])
                    print(f"Distances for first {min_n_atoms} atoms")
                    print(value[0][0:min_n_atoms])
                    print(value[1][0:min_n_atoms])
                raise
            elif key == "edge_index":
                print(f"training vs calculation input: {value[0].shape} vs {value[1].shape}. Differences might result from different handling of cutoff and might be inconsequential.")
            elif key == "cell":
                print(f"training vs calculation input:\n {value[0]} vs {value[1]}\n Differences might result from different starting geometries and might be inconsequential.")
            else:
                print(value[0])
                print(value[1])
                raise
    print("Input Assertions passed")

def run_output_assertions(predicted_charges, predicted_energies, predicted_forces, written_charges, written_energy, written_force, reference_charges, reference_energy, reference_force):
    print("------------------Output assertion-------------------------")
    predicted_charge, predicted_energy, predicted_force = [output[0] for output in [predicted_charges, predicted_energies, predicted_forces]] # outputs of the first model
    
    try:
        assert np.allclose(predicted_charge, written_charges, atol=1e-03), "Charge Assertion failed"
    except AssertionError:
        print("Charge")
        print(predicted_charge)
        print(written_charges)
        if STARTING_STRUCTURE_IDX is not None:
            print(reference_charges)    

        print("Total charge")
        print(np.sum(predicted_charge))
        print(np.sum(written_charges))
        if STARTING_STRUCTURE_IDX is not None:
            print(np.sum(reference_charges))
        raise

    try:
        assert np.allclose(predicted_energy, written_energy, atol=1e-05), "Energy Assertion failed"
    except AssertionError:
        print("Energy")
        print(predicted_energy)
        print(written_energy)
        if STARTING_STRUCTURE_IDX is not None:
            print(reference_energy)
        raise

    try:
        assert np.allclose(predicted_force, written_force, atol=1e-03), "Force Assertion failed"
    except AssertionError:
        print("Force")
        print(predicted_force.reshape(1,-1,3))
        print(written_force.reshape(1,-1,3))
        if STARTING_STRUCTURE_IDX is not None:
            print(reference_force)
        raise
    print("Output Assertions passed")

def run_std_assertions(predicted_energies, predicted_forces, n_atoms):
    print("------------------Std assertion-------------------------")
    print("Predicted shapes:", predicted_energies.shape, predicted_forces.shape)
    print("Written Shapes:", predicted_energies.shape, predicted_forces.shape)
    energy_mean = np.mean(predicted_energies, axis=0).flatten()[0] # shape(n_molecules,)
    energy_std = np.std(predicted_energies, axis=0).flatten()[0] # shape(n_molecules,)
    force_mean = np.mean(predicted_forces, axis=0) # shape(n_molecules,n_atoms,3)
    force_std = np.std(predicted_forces, axis=0) # shape(n_molecules,n_atoms,3)
    qm_mlmm_energy_mean = np.genfromtxt("qm_mlmm_std.xyz", skip_header=2, max_rows=1)
    qm_mlmm_energy_std = np.genfromtxt("qm_mlmm_std.xyz", skip_header=3+2, max_rows=1)
    qm_mlmm_force_mean = np.genfromtxt("qm_mlmm_std.xyz", skip_header=3+3+2, max_rows=n_atoms, usecols=(2,3,4))
    qm_mlmm_force_std = np.genfromtxt("qm_mlmm_std.xyz", skip_header=3+3+(2+n_atoms)+2, max_rows=n_atoms, usecols=(2,3,4))
    try:
        assert np.allclose(energy_mean, qm_mlmm_energy_mean, atol=1e-03), "Energy Mean Assertion failed"
    except AssertionError as e:
        print(e)
        print(energy_mean)
        print(qm_mlmm_energy_mean)
        raise

    try:
        assert np.allclose(energy_std, qm_mlmm_energy_std, atol=1e-03), "Energy Std Assertion failed"
    except AssertionError as e:
        print(e)
        print(energy_std)
        print(qm_mlmm_energy_std)
        raise

    try:
        assert np.allclose(force_mean.reshape(-1,3), qm_mlmm_force_mean, atol=1e-03), "Force Mean Assertion failed"
    except AssertionError as e:
        print(e)
        print(force_mean.reshape(-1,3))
        print(qm_mlmm_force_mean)
        raise

    try:
        assert np.allclose(force_std.reshape(-1,3), qm_mlmm_force_std, atol=1e-03), "Force Std Assertion failed"
    except AssertionError as e:
        print(e)
        print(force_std.reshape(-1,3))
        print(qm_mlmm_force_std)
        raise

    print("Std Assertions passed")

if __name__ == "__main__":
    main()