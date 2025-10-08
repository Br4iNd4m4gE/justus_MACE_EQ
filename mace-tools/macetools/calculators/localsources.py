import torch
from ase.calculators.calculator import Calculator, all_changes

from macetools import data
from mace.tools import torch_geometric, torch_tools, utils
from ase.stress import full_3x3_to_voigt_6_stress


class MACELocalSymmetricCharges(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress", "partial_charges", "partial_dipoles", "dipole"]

    def __init__(
        self,
        model_path: str,
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        charges_key="charges",
        **kwargs,
    ):
        """
        :param charges_key: str, Array field of atoms object where atomic charges are stored
        """
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.device = torch_tools.init_device(device)
        torch_tools.set_default_dtype(default_dtype)

        self.model = torch.load(f=model_path, map_location=self.device)
        self.model = self.model.to(
            self.device
        )

        self.r_max = self.model.r_max
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.charges_key = charges_key


    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        config = data.config_from_atoms(
            atoms,
        )
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=float(self.r_max)
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)

        # predict + extract data
        out = self.model(batch, compute_force=True, compute_stress=True)
        energy = out["energy"].detach().cpu().item()
        forces = out["forces"].detach().cpu().numpy()
        density_coefficients = out["density_coefficients"].detach().cpu().numpy()
        partial_charges = density_coefficients[:, 0]
        partial_dipoles = density_coefficients[:, [3,1,2]]

        dipole = out['dipole'].detach().cpu().numpy()
        stress = full_3x3_to_voigt_6_stress(
            out["stress"].detach().cpu().numpy()[0]
        )

        # store results
        E = energy * self.energy_units_to_eV
        self.results = {
            "energy": E,
            "free_energy": E,
            "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
            "stress" : stress * (self.energy_units_to_eV / self.length_units_to_A**3),
            "partial_charges": partial_charges,
            "partial_dipoles": partial_dipoles,
            "dipole": dipole
        }
