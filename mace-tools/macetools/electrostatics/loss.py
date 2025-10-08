import torch

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch

from mace.modules.loss import (
    weighted_mean_squared_error_energy,
    mean_squared_error_forces,
)



class WeightedChargesEnergyForcesLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, density_coefficients_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "charges_weight",
            torch.tensor(density_coefficients_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.charges_weight * weighted_mean_squared_error_charges(ref, pred))


def weighted_mean_squared_error_dma(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # dma: [n_atoms, many]
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    configs_density_coefficients_weight = torch.repeat_interleave(
        ref.density_coefficients_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    assert ref["density_coefficients"].shape == pred["density_coefficients"].shape
    sq_error = torch.square(
        ref["density_coefficients"] - pred["density_coefficients"]
    )  # [n_nodes, (max_l+1)**2]

    return torch.mean(configs_weight * configs_density_coefficients_weight * sq_error)


def weighted_mean_squared_error_dipole(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # dipole: [n_graphs, 3]
    configs_weight = ref.weight.view(-1, 1)  # [n_graphs, 1]
    configs_dipole_weight = ref.dipole_weight.view(-1, 1)  # [n_graphs, 1]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1)  # [n_graphs, 1]
    return torch.mean(
        configs_weight
        * configs_dipole_weight
        * torch.square((ref["dipole"] - pred["dipole"]) / num_atoms.unsqueeze(-1))
    )


class WeightedChargesLoss(torch.nn.Module):
    def __init__(self, charges_weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer(
            "charges_weight",
            torch.tensor(charges_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return self.charges_weight * weighted_mean_squared_error_charges(ref, pred)

    def __repr__(self):
        return f"{self.__class__.__name__}(charges_weight={self.charges_weight:.3f}"

def weighted_mean_squared_error_charges(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # dma: [n_atoms, many]

    configs_weight = torch.repeat_interleave(ref.weight, ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)  # [n_atoms, 1]
    configs_charges_weight = torch.repeat_interleave(
        ref.charges_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    assert ref["charges"].shape == pred["charges"].shape
    sq_error = torch.square(
        ref["charges"] - pred["charges"]
    )  # [n_nodes, (max_l+1)**2]'
    return torch.mean(configs_weight * configs_charges_weight * sq_error)



class WeightedDensityCoefficientsLoss(torch.nn.Module):
    def __init__(self, density_coefficients_weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer(
            "density_coefficients_weight",
            torch.tensor(density_coefficients_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return self.density_coefficients_weight * weighted_mean_squared_error_dma(ref, pred)

    def __repr__(self):
        return f"{self.__class__.__name__}(DMA_weight={self.density_coefficients_weight:.3f}"


class WeightedEnergyForcesDensityLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, density_coefficients_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "density_coefficients_weight",
            torch.tensor(density_coefficients_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.density_coefficients_weight * weighted_mean_squared_error_dma(ref, pred)
        )


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.dipole_weight * weighted_mean_squared_error_dipole(ref, pred)
        )


class WeightedEnergyForcesDensityDipoleLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        density_coefficients_weight=1.0,
        dipole_weight=1.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "density_coefficients_weight",
            torch.tensor(density_coefficients_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.density_coefficients_weight * weighted_mean_squared_error_dma(ref, pred)
            + self.dipole_weight * weighted_mean_squared_error_dipole(ref, pred)
        )


class WeightedDensityDipoleLoss(torch.nn.Module):
    def __init__(self, density_coefficients_weight=1.0, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "density_coefficients_weight",
            torch.tensor(density_coefficients_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
    
    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.density_coefficients_weight * weighted_mean_squared_error_dma(ref, pred)
            + self.dipole_weight * weighted_mean_squared_error_dipole(ref, pred)
        )
