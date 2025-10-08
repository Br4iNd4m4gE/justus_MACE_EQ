import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch.nn.functional
from e3nn import o3
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_ema import ExponentialMovingAverage

import mace
from mace.tools import torch_geometric

# do not make macetools things called tools and modules, then its easy.
from mace import tools
import mace.modules
from mace.tools.scripts_utils import LRScheduler

# macetools replaces data, and some utils
from macetools import data
import macetools.utils

from macetools import electrostatics

def load_dataset(args: argparse.Namespace) -> Tuple[macetools.utils.script_utils.SubsetCollection, torch_geometric.dataloader.DataLoader, torch_geometric.dataloader.DataLoader, torch_geometric.dataloader.DataLoader, tools.AtomicNumberTable, np.ndarray]:
    """Load dataset from xyz files

    Args:
        args (argparse.Namespace): Namespace object containing arguments

    Returns:
        Sequence[macetools.utils.script_utils.SubsetCollection, torch_geometric.dataloader.DataLoader, torch_geometric.dataloader.DataLoader, torch_geometric.dataloader.DataLoader, tools.AtomicNumberTable, np.ndarray]: Tuple containing collections, train_loader, valid_loader, test_loader, z_table, atomic_energies
    """

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    # Data preparation
    collections, atomic_energies_dict = macetools.utils.get_dataset_from_xyz(
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=args.test_file,
        seed=args.valid_set_seed,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        virials_key=args.virials_key,
        dipole_key=args.dipole_key,
        charges_key=args.charges_key,
        density_coefficients_key=args.atomic_multipoles_key,
        density_coefficients_max_l=args.atomic_multipoles_max_l,
        total_charge_key=args.total_charge_key,
        external_field_key=args.external_field_key,
        fermi_level_key=args.fermi_level_key,
        esp_key=args.esp_key,
        esp_gradient_key=args.esp_gradient_key,
    )

    logging.info(
        f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
        f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}]"
    )

    # Atomic number table
    z_table = tools.get_atomic_number_table_from_zs(
        z
        for configs in (collections.train, collections.valid)
        for config in configs
        for z in config.atomic_numbers
    )
    logging.info(z_table)

    # energies
    if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
        if args.E0s is not None:
            logging.info(
                "Atomic Energies not in training file, using command line argument E0s"
            )
            if args.E0s.lower() == "average":
                logging.info(
                    "Computing average Atomic Energies using least squares regression"
                )
                atomic_energies_dict = data.compute_average_E0s(
                    collections.train, z_table
                )
            else:
                try:
                    atomic_energies_dict = ast.literal_eval(args.E0s)
                    assert isinstance(atomic_energies_dict, dict)
                except Exception as e:
                    raise RuntimeError(
                        f"E0s specified invalidly, error {e} occured"
                    ) from e
        else:
            raise RuntimeError(
                "E0s not found in training file and not specified in command line"
            )
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        logging.info(f"Atomic energies: {atomic_energies.tolist()}")

    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in collections.train
        ],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in collections.valid
        ],
        batch_size=args.valid_batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for name, test_configs in collections.tests # Entries in collections.tests are tuples of (name, [configs]), need configs here
            for config in test_configs 
        ],
        batch_size=args.valid_batch_size,
        shuffle=False,
        drop_last=False,
    )

    return collections, train_loader, valid_loader, test_loader, z_table, atomic_energies

def load_model(model_path: str, device: str) -> torch.nn.Module:
        try:
            model = torch.jit.load(f=model_path, map_location=device)
        except RuntimeError as e:
            logging.warning("Model file is not a torchscript .pt file, trying to load as a .model file")
            try:
                model = torch.load(f=model_path, map_location=device)
            except FileNotFoundError as e:
                logging.error(f"Model file not found: {e}")
                raise            
        except FileNotFoundError as e:
            logging.error(f"Model file not found: {e}")
            raise

        model = model.to(device)  # shouldn't be necessary but seems to help with CUDA problems
        return model