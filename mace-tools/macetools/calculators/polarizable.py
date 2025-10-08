import torch
from ase.calculators.calculator import Calculator, all_changes


from macetools import data

from mace.tools import torch_tools, utils, torch_geometric
import time


class MACEPolarizable(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "charges"]

    def __init__(
        self,
        model_path: str,
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype: str = "float64",
        formal_charges_key: str = "charges",
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        self.results = {}

        self.device = torch_tools.init_device(device)
        torch_tools.set_default_dtype(default_dtype)

        self.model = torch.load(f=model_path, map_location=self.device).to(self.device)

        self.r_max = self.model.r_max

        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.formal_charges_key = formal_charges_key

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)

        config = data.config_from_atoms(atoms)
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
        ref_charges = torch.tensor(batch[self.formal_charges_key], dtype=torch.float64)
        total_charge = torch.sum(ref_charges)
        t1 = time.time()
        out = self.model(
            batch,
            compute_force=True,
            num_scf_steps=3,
            target_charge=total_charge,
            training=False,
        )
        t2 = time.time()

        print("Time for force call", t2 - t1)

        energy = out["energy"].detach().cpu().item()
        forces = out["forces"].detach().cpu().numpy()
        charges = out["density_coefficients"].detach().cpu().numpy()

        # store results
        E = energy * self.energy_units_to_eV
        self.results = {
            "energy": E,
            "free_energy": E,
            # force has units eng / len:
            "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
            # stress has units eng / len:
            "charges": charges,
        }
