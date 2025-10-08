###########################################################################################
# Training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import ast
import json
import logging
from pathlib import Path
from typing import Optional

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


def get_param_options(model, args, decay_interactions, no_decay_interactions):
    param_options = dict(
        params=[
            {
                "name": "embedding",
                "params": model.node_embedding.parameters(),
                "weight_decay": 0.0,
            },
            {
                "name": "interactions_decay",
                "params": list(decay_interactions.values()),
                "weight_decay": args.weight_decay,
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "products",
                "params": model.products.parameters(),
                "weight_decay": args.weight_decay,
            },
            {
                "name": "readouts",
                "params": model.readouts.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
        amsgrad=args.amsgrad,
    )

    if args.model == "LocalSymmetricCharges":
        param_options["params"].append(
            {
                "name": "lr_source_maps",
                "params": model.lr_source_maps.parameters(),
                "weight_decay": 0.0,
                "lr": 0.01,
            }
        )
    if args.model == "NonPolarizable":
        param_options["params"].append(
            {
                "name": "lr_source_maps",
                "params": model.lr_source_maps.parameters(),
                "weight_decay": 0.0,
            }
        )
    if args.model == "DFTbaselined":
        param_options["params"].append(
            {
                "name": "charge_coefs_linears",
                "params": model.charge_coefs_linears.parameters(),
                "weight_decay": 0.0,
            }
        )
        param_options["params"].append(
            {
                "name": "charge_readouts",
                "params": model.charge_readouts.parameters(),
                "weight_decay": 0.0,
            }
        )
    if args.model == "Polarizable":
        param_options["params"].append(
            {
                "name": "lr_source_maps",
                "params": model.lr_source_maps.parameters(),
                "weight_decay": 0.0,
            }
        )
        param_options["params"].append(
            {
                "name": "field_dependent_charges_map",
                "params": model.field_dependent_charges_map.parameters(),
                "weight_decay": args.weight_decay,
            }
        )
        param_options["params"].append(
            {
                "name": "local_electron_energy",
                "params": model.local_electron_energy.parameters(),
                "weight_decay": args.weight_decay,
            }
        )
    if args.model == "SymmetricPolarizable":
        param_options["params"].append(
            {
                "name": "lr_source_maps",
                "params": model.lr_source_maps.parameters(),
                "weight_decay": 0.0,
                "lr": 0.01,
            }
        )
        param_options["params"].append(
            {
                "name": "field_dependent_charges_map",
                "params": model.field_dependent_charges_map.parameters(),
                "weight_decay": args.weight_decay,
                "lr": 0.01,
            }
        )

    return param_options


def main() -> None:
    args = macetools.utils.extended_arg_parser().parse_args()
    tag = tools.get_tag(name=args.name, seed=args.seed)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.info(f"Configuration: {args}")
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    if args.valid_set_seed is None:
        args.valid_set_seed = args.seed

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

    # choices to simply this repo:
    compute_energy = True
    assert args.compute_forces
    assert not (args.compute_stress)
    compute_virials = False
    compute_dipole = False

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
        drop_last=True,
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

    # loss
    loss_fn: torch.nn.Module
    if args.loss == "energy_forces":
        loss_fn = mace.modules.WeightedEnergyForcesLoss(
            energy_weight=args.energy_weight, forces_weight=args.forces_weight
        )
    if args.loss == "atomic_multipoles":
        loss_fn = electrostatics.WeightedDensityCoefficientsLoss(
            density_coefficients_weight=args.atomic_multipoles_weight,
        )
    elif args.loss == "energy_forces_atomic_multipoles":
        loss_fn = electrostatics.WeightedEnergyForcesDensityLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            density_coefficients_weight=args.atomic_multipoles_weight,
        )
    elif args.loss == "energy_forces_dipole":
        loss_fn = electrostatics.WeightedEnergyForcesDipoleLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            dipole_weight=args.dipole_weight,
        )
    elif args.loss == "energy_forces_dipole_atomic_multipoles":
        loss_fn = electrostatics.WeightedEnergyForcesDensityDipoleLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            density_coefficients_weight=args.atomic_multipoles_weight,
            dipole_weight=args.dipole_weight,
        )
    elif args.loss == "atomic_multipoles_dipole":
        loss_fn = electrostatics.WeightedDensityDipoleLoss(
            density_coefficients_weight=args.atomic_multipoles_weight,
            dipole_weight=args.dipole_weight,
        )
    else:
        loss_fn = mace.modules.WeightedEnergyForcesLoss(
            energy_weight=args.energy_weight, forces_weight=args.forces_weight
        )
    logging.info(loss_fn)

    if args.compute_avg_num_neighbors:
        args.avg_num_neighbors = mace.modules.compute_avg_num_neighbors(train_loader)
    logging.info(f"Average number of neighbors: {args.avg_num_neighbors}")

    # Selecting outputs     # leave here incase we find a problem
    # compute_virials = False
    # if args.loss in ("stress", "virials", "huber"):
    #    compute_virials = True
    #    args.compute_stress = True
    #    args.error_table = "PerAtomRMSEstressvirials"

    output_args = {
        "energy": compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": args.compute_stress,
    }
    logging.info(f"Selected the following outputs: {output_args}")

    # Build model
    logging.info("Building model")
    if args.num_channels is not None and args.max_L is not None:
        assert args.num_channels > 0, "num_channels must be positive integer"
        assert args.max_L >= 0, "max_L must be non-negative integer"
        args.hidden_irreps = o3.Irreps(
            (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
            .sort()
            .irreps.simplify()
        )

    assert (
        len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
    ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

    if args.formal_charges_from_data:
        logging.info("charged models: taking formal charges per config from data")
        atomic_charges = None
    elif args.atomic_formal_charges is not None:
        logging.info("charged models: taking formal charges from command line")
        atomic_charges_dict = ast.literal_eval(args.atomic_formal_charges)
        assert isinstance(atomic_charges_dict, dict)
        atomic_charges: np.ndarray = np.array(
            [atomic_charges_dict[z] for z in z_table.zs]
        )

    logging.info(f"Hidden irreps: {args.hidden_irreps}")
    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=mace.modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
        correlation=args.correlation,
        gate=mace.modules.gate_dict[args.gate],
        MLP_irreps=o3.Irreps(args.MLP_irreps),
        radial_MLP=ast.literal_eval(args.radial_MLP),
        radial_type=args.radial_type,
    )

    model: torch.nn.Module

    if args.model == "MACE":
        if args.scaling == "no_scaling":
            std = 1.0
            logging.info("No scaling selected")
        else:
            mean, std = mace.modules.scaling_classes[args.scaling](
                train_loader, atomic_energies
            )
        model = mace.modules.ScaleShiftMACE(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            atomic_inter_scale=std,
            atomic_inter_shift=0.0,
        )
    elif args.model == "ScaleShiftMACE":
        mean, std = mace.modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
        )
        model = mace.modules.ScaleShiftMACE(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                args.interaction_first
            ],
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
        )
    elif args.model == "FixedChargeBaselinedMACE":
        formal_charges = ast.literal_eval(args.atomic_formal_charges)
        assert len(formal_charges) == len(z_table)
        model = electrostatics.FixedChargeBaselinedMACE(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                args.interaction_first
            ],
            kspace_cutoff_factor=args.kspace_cutoff_factor,
            atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
            atomic_formal_charges=atomic_charges,
            formal_charges_from_data=args.formal_charges_from_data,
        )
    elif args.model == "LocalSymmetricCharges":
        formal_charges = ast.literal_eval(args.atomic_formal_charges)
        assert len(formal_charges) == len(z_table)
        model = electrostatics.LocalSymmetricCharges(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                args.interaction_first
            ],
            kspace_cutoff_factor=args.kspace_cutoff_factor,
            atomic_multipoles_max_l=args.atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
            atomic_formal_charges=atomic_charges,
            formal_charges_from_data=args.formal_charges_from_data,
            include_electrostatic_self_interaction=args.include_electrostatic_self_interaction,
        )
    elif args.model == "NonPolarizable":
        model = electrostatics.NonPolarizable(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                args.interaction_first
            ],
            kspace_cutoff_factor=args.kspace_cutoff_factor,
            atomic_multipoles_max_l=args.atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
            add_coulomb=not args.dont_add_coulomb,
            include_electrostatic_self_interaction=args.include_electrostatic_self_interaction,
        )
    elif args.model == "DFTbaselined":
        if args.scaling == "no_scaling":
            std = 1.0
            logging.info("No scaling selected")
        else:
            mean, std = mace.modules.scaling_classes[args.scaling](
                train_loader, atomic_energies
            )
        model = electrostatics.DFTbaselined(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            atomic_inter_scale=std,
            atomic_inter_shift=0.0,
            kspace_cutoff_factor=args.kspace_cutoff_factor,
            atomic_multipoles_max_l=args.atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
            include_pbc_corrections=not(args.dont_include_pbc_corrections),
            include_electrostatic_self_interaction=args.include_electrostatic_self_interaction
        )
    elif args.model == "ElectrostaticsEvaluator":
        model = electrostatics.ElectrostaticsEvaluator(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            atomic_inter_scale=std,
            atomic_inter_shift=0.0,
            kspace_cutoff_factor=args.kspace_cutoff_factor,
            atomic_multipoles_max_l=args.atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
        )
    elif args.model == "Polarizable":
        model = electrostatics.Polarizable(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                args.interaction_first
            ],
            kspace_cutoff_factor=args.kspace_cutoff_factor,
            atomic_multipoles_max_l=args.atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
            field_feature_widths=ast.literal_eval(args.field_feature_widths),
            include_electrostatic_self_interaction=args.include_electrostatic_self_interaction,
            add_local_electron_energy=args.include_local_electron_energy,
            field_dependence_type=args.field_dependence_type,
            final_field_readout_type=args.final_field_readout_type,
            quadrupole_feature_corrections=args.quadrupole_feature_corrections
        )
    elif args.model == "SymmetricPolarizable":
        model = electrostatics.SymmetricPolarizable(
            **model_config,
            interaction_cls_first=mace.modules.interaction_classes[
                args.interaction_first
            ],
            kspace_cutoff_factor=args.kspace_cutoff_factor,
            atomic_multipoles_max_l=args.atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=args.atomic_multipoles_smearing_width,
            field_feature_widths=ast.literal_eval(args.field_feature_widths),
            atomic_formal_charges=atomic_charges,
            formal_charges_from_data=args.formal_charges_from_data,
        )
    else:
        raise RuntimeError(f"Unknown model: '{args.model}'")

    model.to(device)

    # Optimizer
    decay_interactions = {}
    no_decay_interactions = {}
    for name, param in model.interactions.named_parameters():
        if "linear.weight" in name or "skip_tp_full.weight" in name:
            decay_interactions[name] = param
        else:
            no_decay_interactions[name] = param

    param_options = get_param_options(
        model, args, decay_interactions, no_decay_interactions
    )

    # parse special options for scf model training
    if args.scf_training_options is not None:
        scf_training_options = ast.literal_eval(args.scf_training_options)
    else:
        scf_training_options = None

    optimizer: torch.optim.Optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)

    logger = tools.MetricsLogger(directory=args.results_dir, tag=tag + "_train")

    lr_scheduler = LRScheduler(optimizer, args)

    swa: Optional[tools.SWAContainer] = None
    swas = [False]
    if args.swa:
        swas.append(True)
        if args.start_swa is None:
            args.start_swa = (
                args.max_num_epochs // 4 * 3
            )  # if not set start swa at 75% of training

        if not("energy" in args.loss):
            logging.info("Can not select swa without energy in loss.")
        elif args.loss == "energy_forces":
            loss_fn_energy = mace.modules.WeightedEnergyForcesLoss(
                args.swa_energy_weight, forces_weight=args.swa_forces_weight
            )
            logging.info(
                f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight}, dipole weight : {args.swa_dipole_weight} and learning rate : {args.swa_lr}"
            )
        elif args.loss == "energy_forces_atomic_multipoles":
            loss_fn_energy = electrostatics.WeightedEnergyForcesDensityLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                density_coefficients_weight=args.swa_atomic_multipoles_weight,
            )
            logging.info(
                f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight}, dipole weight : {args.swa_dipole_weight} and learning rate : {args.swa_lr}"
            )
        elif args.loss == "energy_forces_dipole":
            loss_fn_energy = electrostatics.WeightedEnergyForcesDipoleLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                dipole_weight=args.swa_dipole_weight,
            )
            logging.info(
                f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight}, dipole weight : {args.swa_dipole_weight} and learning rate : {args.swa_lr}"
            )
        elif args.loss == "energy_forces_dipole_atomic_multipoles":
            loss_fn_energy = electrostatics.WeightedEnergyForcesDensityDipoleLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                density_coefficients_weight=args.swa_atomic_multipoles_weight,
                dipole_weight=args.swa_dipole_weight,
            )
            logging.info(
                f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight}, dipole weight : {args.swa_dipole_weight} and learning rate : {args.swa_lr}"
            )
        elif args.loss == "atomic_multipoles_dipole":
            loss_fn_energy = electrostatics.WeightedDensityDipoleLoss(
                density_coefficients_weight=args.swa_atomic_multipoles_weight,
                dipole_weight=args.swa_dipole_weight,
            )
            logging.info(
                f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight}, dipole weight : {args.swa_dipole_weight} and learning rate : {args.swa_lr}"
            )
        swa = tools.SWAContainer(
            model=AveragedModel(model),
            scheduler=SWALR(
                optimizer=optimizer,
                swa_lr=args.swa_lr,
                anneal_epochs=1,
                anneal_strategy="linear",
            ),
            start=args.start_swa,
            loss_fn=loss_fn_energy,
        )

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=args.start_swa,
    )

    start_epoch = 0
    if args.restart_latest:
        try:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=True,
                device=device,
            )
        except Exception as e:  # pylint: disable=W0703
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)

    logging.info(model)
    logging.info(f"Number of parameters: {tools.count_parameters(model)}")
    logging.info(f"Optimizer: {optimizer}")

    if args.wandb:
        logging.info("Using Weights and Biases for logging")
        import wandb

        wandb_config = {}
        args_dict = vars(args)
        args_dict_json = json.dumps(args_dict)
        for key in args.wandb_log_hypers:
            wandb_config[key] = args_dict[key]
        tools.init_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=wandb_config,
        )
        wandb.run.summary["params"] = args_dict_json

    macetools.utils.train(  # this is the macetools train
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        output_args=output_args,
        device=device,
        swa=swa,
        ema=ema,
        max_grad_norm=args.clip_grad,
        log_errors=args.error_table,
        log_wandb=args.wandb,
        scf_training_options=scf_training_options,
        use_pbc_evaluator=args.use_pbc_evaluator
    )

    # Evaluation on test datasets
    logging.info("Computing metrics for training, validation, and test sets")

    all_collections = [
        ("train", collections.train),
        ("valid", collections.valid),
    ] + collections.tests
    
    for swa_eval in swas:
        epoch = checkpoint_handler.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler),
            swa=swa_eval,
            device=device,
        )
        model.to(device)
        logging.info(f"Loaded model from epoch {epoch}")

        for param in model.parameters():
            param.requires_grad = False
        table = macetools.utils.create_error_table(
            table_type=args.error_table,
            all_collections=all_collections,
            z_table=z_table,
            r_max=args.r_max,
            valid_batch_size=args.valid_batch_size,
            model=model,
            loss_fn=loss_fn,
            output_args=output_args,
            log_wandb=args.wandb,
            device=device,
            scf_training_options=scf_training_options,
            use_pbc_evaluator=args.use_pbc_evaluator
        )
        logging.info("\n" + str(table))

        # Save entire model
        if swa_eval:
            model_path = Path(args.checkpoints_dir) / (tag + "_swa.model")
        else:
            model_path = Path(args.checkpoints_dir) / (tag + ".model")
        logging.info(f"Saving model to {model_path}")
        if args.save_cpu:
            model = model.to("cpu")
        torch.save(model, model_path)

        if swa_eval:
            torch.save(model, Path(args.model_dir) / (args.name + "_swa.model"))
        else:
            torch.save(model, Path(args.model_dir) / (args.name + ".model"))

    # now save all checkpoints as models:
    get_all_checkpoint_paths = checkpoint_handler.io.get_all_checkpoint_paths(swa_only=True)

    for checkpoint_path in get_all_checkpoint_paths:
        epoch = checkpoint_handler.load(
            state=tools.CheckpointState(model, optimizer, lr_scheduler),
            path=checkpoint_path,
            device=device,
        )
        model.to(device)
        logging.info(f"Loaded model from epoch {epoch}")

        for param in model.parameters():
            param.requires_grad = False

        # Save entire model
        if swa_eval:
            model_path = Path(args.checkpoints_dir) / (tag + "_swa.model")
        else:
            model_path = Path(args.checkpoints_dir) / (tag + ".model")
        
        logging.info(f"Saving model to {model_path}")
        torch.save(model, checkpoint_path[:-3] + ".model")

    logging.info("Done")


if __name__ == "__main__":
    main()
