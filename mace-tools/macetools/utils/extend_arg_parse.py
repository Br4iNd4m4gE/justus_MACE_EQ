import argparse
from typing import Optional
from mace.tools import build_default_arg_parser


def extended_arg_parser() -> argparse.ArgumentParser:
    parser = build_default_arg_parser()

    # new arguments
    parser.add_argument(
        "--charges_weight",
        help="weight of charges loss",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--swa_charges_weight",
        help="weight of charges loss after starting swa",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--scale_atsize",
        help="How to scale atomic sizes",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--atomic_multipoles_key",
        help="Key of density coefficients_key in training xyz",
        type=str,
        default="density_coefficients",
    )
    parser.add_argument(
        "--total_charge_key",
        help="Key of density coefficients_key in training xyz",
        type=str,
        default="total_charge",
    )
    parser.add_argument(
        "--atomic_multipoles_weight",
        help="weights of local density coefficients",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--swa_atomic_multipoles_weight",
        help="weights of local density coefficients",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--external_field_key",
        type=str,
        default="external_field",
    )
    parser.add_argument(
        "--fermi_level_key",
        type=str,
        default="fermi_level",
    )
    parser.add_argument(
        "--esp_key",
        type=str,
        default="esp",
    )
    parser.add_argument(
        "--esp_gradient_key",
        type=str,
        default="esp_gradient",
    )
    parser.add_argument(
        "--dont_add_coulomb",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--include_local_electron_energy",
        action="store_true",
        default=False
    )
    parser.add_argument("--atomic_multipoles_max_l", type=int, default=0)
    parser.add_argument("--atomic_multipoles_smearing_width", type=float, default=1.0)
    parser.add_argument("--kspace_cutoff_factor", type=float, default=1.5)
    parser.add_argument("--atomic_formal_charges", type=str)
    parser.add_argument(
        "--formal_charges_from_data", action="store_true", default=False
    )
    parser.add_argument("--field_feature_widths", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--dont_include_pbc_corrections", action="store_true"
    )
    parser.add_argument(
        "--include_electrostatic_self_interaction", action="store_true"
    )
    parser.add_argument(
        "--valid_set_seed", type=int, default=None
    )
    parser.add_argument(
        "--scf_training_options", type=str, default=None
    )
    parser.add_argument(
        "--field_dependence_type", type=str, default="local_linear"
    )
    parser.add_argument(
        "--final_field_readout_type", type=str, default="StrictQuadraticFieldEnergyReadout"
    )
    parser.add_argument(
        "--quadrupole_feature_corrections", action="store_true", default=False
    )
    parser.add_argument(
        "--use_pbc_evaluator", action="store_true", default=False
    )
    

    # update existing arguments
    for item in parser._actions:
        if item.option_strings == ["--loss"]:
            item.choices = [
                "atomic_multipoles",
                "energy_forces",
                "energy_forces_atomic_multipoles",
                "energy_forces_dipole",
                "energy_forces_dipole_atomic_multipoles",
                "atomic_multipoles_dipole",
                "charges",
                "atomic_multipoles",
                "energy",
                "charges_energy_forces",
            ]
        elif item.option_strings == ["--error_table"]:
            item.choices = [
                "PerAtomRMSE",
                "DensityCoefficientsRMSE",
                "DensityEnergyRMSE",
                "DipoleRMSE",
                "DensityDipoleRMSE",
                "ChargesRMSE",
                "EFQRMSE"
            ]
        elif item.option_strings == ["--model"]:
            item.choices = [
                "ScaleShiftMACE",
                "MACE",
                "QEq",
                "maceQEq",
                "maceQEqESP",
                "Lqeq",
                "LocalSymmetricCharges",
                "Polarizable",
                "NonPolarizable",
                "SymmetricPolarizable",
                "DFTbaselined",
                "FixedChargeBaselinedMACE"
            ]

    return parser
