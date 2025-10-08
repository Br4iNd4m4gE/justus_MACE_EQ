import dataclasses
import logging
from typing import Dict, List, Optional, Tuple, Iterable

import torch
from prettytable import PrettyTable

from mace.tools import AtomicNumberTable, torch_geometric

# update
from .. import data
from ..data import AtomicData
from .train import evaluate

@dataclasses.dataclass
class SubsetCollection:
    train: data.Configurations
    valid: data.Configurations
    tests: List[Tuple[str, data.Configurations]]


def get_dataset_from_xyz(
    train_path: str,
    valid_path: str,
    valid_fraction: float,
    config_type_weights: Dict,
    test_path: str = None,
    seed: int = 1234,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipoles",
    charges_key: str = "charges",
    density_coefficients_key: str = "density_coefficients",
    total_charge_key: str = "total_charge",
    external_field_key: str = "external_field",
    fermi_level_key: str = "fermi_level",
    esp_key: str = "esp",
    esp_gradient_key: str = "esp_gradient",
    density_coefficients_max_l: int = 0,
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from xyz file"""
    atomic_energies_dict, all_train_configs = data.load_from_xyz(
        file_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        density_coefficients_key=density_coefficients_key,
        density_coefficients_max_l=density_coefficients_max_l,
        total_charge_key=total_charge_key,
        external_field_key=external_field_key,
        fermi_level_key=fermi_level_key,
        esp_key=esp_key,
        esp_gradient_key=esp_gradient_key,
        extract_atomic_energies=True,
    )
    logging.info(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )
    if valid_path is not None:
        _, valid_configs = data.load_from_xyz(
            file_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            density_coefficients_key=density_coefficients_key,
            density_coefficients_max_l=density_coefficients_max_l,
            total_charge_key=total_charge_key,
            external_field_key=external_field_key,
            fermi_level_key=fermi_level_key,
            esp_key=esp_key,
            esp_gradient_key=esp_gradient_key,
            extract_atomic_energies=False,
        )
        logging.info(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
        )
        train_configs = all_train_configs
    else:
        logging.info(
            "Using random %s%% of training set for validation", 100 * valid_fraction
        )
        train_configs, valid_configs = data.random_train_valid_split(
            all_train_configs, valid_fraction, seed
        )

    test_configs = []
    if test_path is not None:
        _, all_test_configs = data.load_from_xyz(
            file_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            density_coefficients_key=density_coefficients_key,
            density_coefficients_max_l=density_coefficients_max_l,
            total_charge_key=total_charge_key,
            external_field_key=external_field_key,
            fermi_level_key=fermi_level_key,
            esp_key=esp_key,
            esp_gradient_key=esp_gradient_key,
            extract_atomic_energies=False,
        )
        # create list of tuples (config_type, list(Atoms))
        test_configs = data.test_config_types(all_test_configs)
        logging.info(
            f"Loaded {len(all_test_configs)} test configurations from '{test_path}'"
        )
    return (
        SubsetCollection(train=train_configs, valid=valid_configs, tests=test_configs),
        atomic_energies_dict,
    )


NEW_TABLE_TYPES = [
    "DensityCoefficientsRMSE", 
    "DensityEnergyRMSE", 
    "PerAtomRMSE",
    "DipoleRMSE",
    "DensityDipoleRMSE",
    "EFQRMSE"
]


def create_error_table(
    table_type: str,
    all_collections: list,
    z_table: AtomicNumberTable,
    r_max: float,
    valid_batch_size: int,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    output_args: Dict[str, bool],
    log_wandb: bool,
    device: str,
    scf_training_options: Optional[Dict[str, str]] = None,
    use_pbc_evaluator: bool = False,
) -> PrettyTable:
    assert table_type in NEW_TABLE_TYPES

    if log_wandb:
        import wandb
    table = PrettyTable()

    if table_type == "DensityCoefficientsRMSE":
        table.field_names = [
            "config_type", 
            "RMSE DMA / e A^l", 
            "rel DMA %",
            "RMSE qs",
            "RMSE dipoles"
        ]
    elif table_type == "DensityEnergyRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE DMA / e A^l",
        ]
    elif table_type == "PerAtomRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
        ]
    elif table_type == "DipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
        ]
    elif table_type == "DensityDipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
            "RMSE DMA / e A^l", 
            "rel DMA %",
        ]
    # add new tables here...

    # copied from mace...
    for name, subset in all_collections:
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
                for config in subset
            ],
            batch_size=valid_batch_size,
            shuffle=False,
            drop_last=False,
        )
        logging.info(f"Evaluating {name} ...")
        _, metrics = evaluate(
            model,
            loss_fn=loss_fn,
            data_loader=data_loader,
            output_args=output_args,
            device=device,
            scf_training_options=scf_training_options,
            use_pbc_evaluator=use_pbc_evaluator,
        )
        if log_wandb:
            wandb_log_dict = {
                name
                + "_final_rmse_e_per_atom": metrics["rmse_e_per_atom"]
                * 1e3,  # meV / atom
                name + "_final_rmse_f": metrics["rmse_f"] * 1e3,  # meV / A
                name + "_final_rel_rmse_f": metrics["rel_rmse_f"],
            }
            wandb.log(wandb_log_dict)

        # add new tables here...
        if table_type == "DensityCoefficientsRMSE":
            if not 'rmse_local_dipoles' in metrics.keys():
                metrics['rmse_local_dipoles'] = 0.0
            
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_dma'] * 1000:.1f}",
                    f"{metrics['rel_rmse_dma']:.1f}",
                    f"{metrics['rmse_charges'] * 1000:.1f}",
                    f"{metrics['rmse_local_dipoles'] * 1000:.1f}",
                ]
            )
        elif table_type == "DensityEnergyRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:.1f}",
                    f"{metrics['rmse_f'] * 1000:.1f}",
                    f"{metrics['rel_rmse_f']:.2f}",
                    f"{metrics['rmse_dma'] * 1000:.1f}",
                ]
            )
        elif table_type == "PerAtomRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:.1f}",
                    f"{metrics['rmse_f'] * 1000:.1f}",
                    f"{metrics['rel_rmse_f']:.2f}",
                ]
            )
        elif table_type == "DipoleRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_mu_per_atom'] * 1000:.1f}",
                    f"{metrics['rel_rmse_mu']:.1f}",
                ]
            )
        elif table_type == "DensityDipoleRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_mu_per_atom'] * 1000:.1f}",
                    f"{metrics['rel_rmse_mu']:.1f}",
                    f"{metrics['rmse_dma'] * 1000:.1f}",
                    f"{metrics['rel_rmse_dma']:.1f}",
                ]
            )
        elif table_type == "EFQRMSE":
            table.field_names = [
                "config_type",
                "RMSE E / meV / atom",
                "RMSE F / meV / A",
                "relative F RMSE %",
                "RMSE q"
            ]
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:.1f}",
                    f"{metrics['rmse_f'] * 1000:.1f}",
                    f"{metrics['rel_rmse_f']:.2f}",
                    f"{metrics['rmse_q'] * 1000:.1f}",
                ]
            )
        # add new tables here...

    return table
