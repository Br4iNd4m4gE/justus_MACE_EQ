""" implementation of local source models, in which no scf or minimization is needed """
import torch
from typing import Any, Callable, Dict, List, Optional, Type
from e3nn.util.jit import compile_mode
from e3nn import o3
import numpy as np

from mace.tools.scatter import scatter_sum, scatter_mean
from mace.modules.utils import (
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)
from graph_longrange.kspace import (
    compute_k_vectors
)
from graph_longrange.gto_electrostatics import (
    PBCAgnosticDirectElectrostaticEnergyBlock,
    gto_basis_kspace_cutoff,
    GTOExternalFieldBlock
)
from graph_longrange.utils import permute_to_e3nn_convention

from mace.modules import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
)
from .field_blocks import (
    NoFieldSymmetricPredictionSourceBlock,
    PerSpeciesFormalChargesBlock,
    PerAtomFormalChargesBlock,
    EnvironmentDependentSourceBlock,
    LinearFieldSymmetricSourceBlock,
    PureElectrostaticsLRFeatureBlock
)
from .utils import compute_total_charge_dipole, compute_polarization


@compile_mode("script")
class SymmetricPolarizable(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        formal_charges_from_data: bool,  # this must be explicit
        gate: Optional[Callable],
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        atomic_formal_charges: Optional[np.ndarray] = None,
        kspace_cutoff_factor: float = 1.5,
        atomic_multipoles_max_l: int = 0,
        atomic_multipoles_smearing_width: float = 1.0,
        field_feature_widths: List[float] = [1.0],
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff( # new
            [atomic_multipoles_smearing_width] + field_feature_widths,
            atomic_multipoles_max_l
        )
        self.register_buffer(
            "kspace_cutoff", torch.tensor(kspace_cutoff, dtype=torch.get_default_dtype()) # new
        )

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        # field setup - just 1/r convolution
        self.electric_potential_descriptor = PureElectrostaticsLRFeatureBlock(
            density_max_l=atomic_multipoles_max_l, 
            density_smearing_width=atomic_multipoles_smearing_width, 
            projection_max_l=atomic_multipoles_max_l, 
            projection_smearing_widths=field_feature_widths,
            kspace_cutoff=kspace_cutoff
        )
        self.external_field_contribution = GTOExternalFieldBlock(
            atomic_multipoles_max_l,
            field_feature_widths,
            "receiver"
        )

        # charge prediction
        if formal_charges_from_data:
            self.formal_charges = PerAtomFormalChargesBlock()
        else:
            self.formal_charges = PerSpeciesFormalChargesBlock(atomic_formal_charges)

        # field dependant charges. TP between node features and local field features. 
        self.charges_irreps = o3.Irreps.spherical_harmonics(atomic_multipoles_max_l)
        lr_sh_irreps = o3.Irreps.spherical_harmonics(atomic_multipoles_max_l)
        potential_irreps = (lr_sh_irreps * len(field_feature_widths)).sort()[0].simplify()

        self.lr_source_maps = torch.nn.ModuleList(
            [
                NoFieldSymmetricPredictionSourceBlock(
                    node_feats_irreps=hidden_irreps,
                    edge_attrs_irreps=sh_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    target_irreps=interaction_irreps,
                    max_l=atomic_multipoles_max_l
                )
            ]
        )

        for i in range(num_interactions - 1):
            hidden_irreps_out = hidden_irreps  # note change

            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))

            self.lr_source_maps.append(
                NoFieldSymmetricPredictionSourceBlock(
                    node_feats_irreps=hidden_irreps,
                    edge_attrs_irreps=sh_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    target_irreps=interaction_irreps,
                    max_l=atomic_multipoles_max_l
                )
            )

        # field dependent charges block
        self.field_dependent_charges_map = LinearFieldSymmetricSourceBlock(
            node_feats_irreps=hidden_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            charges_irreps=self.charges_irreps,
            field_feats_irreps=potential_irreps,
            target_irreps=interaction_irreps,
        )

        # electric energy
        self.coulomb_energy = PBCAgnosticDirectElectrostaticEnergyBlock(
            density_max_l=atomic_multipoles_max_l,
            density_smearing_width=atomic_multipoles_smearing_width,
            kspace_cutoff=kspace_cutoff,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        external_field: Optional[torch.Tensor] = None,
        num_scf_steps: int = 0,
        constant_charge: Optional[torch.Tensor] = None,
        mixing_parameter: float = 1.0,
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert not any((compute_virials, compute_stress))

        # Setup
        if not training:
            for p in self.parameters():
                p.requires_grad = False
        else:
            for p in self.parameters():
                p.requires_grad = True

        data["node_attrs"].requires_grad_(False)
        if compute_force:
            data["positions"].requires_grad_(True)
        else:
            data["positions"].requires_grad_(False)
        
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(permute_to_e3nn_convention(vectors))
        edge_feats = self.radial_embedding(lengths)

        # -- new bits  --
        if external_field is None:
            external_field = torch.zeros(
                (num_graphs, 4),
                dtype=torch.get_default_dtype(),
                device=data["positions"].device,
            )

        # density
        charge_density = torch.zeros(
            (data["batch"].size(-1), self.charges_irreps.dim),
            device=data["batch"].device,
            dtype=torch.get_default_dtype(),
        )

        # kgrid - note that stress doesn't work yet sicne you need the der. w.r.t. rcell
        k_vectors, kv_norms_squared, kv_mask = compute_k_vectors(
            self.kspace_cutoff, data["cell"].view(-1,3,3), data["rcell"].view(-1,3,3)
        )

        # fixed formal charges
        charge_density[:, 0] += self.formal_charges(data["node_attrs"], data["charges"])

        # formal charges dipole
        edge_fluxes = torch.zeros_like(lengths)  # [n_edges, 1]
        FQ = self.formal_charges(data["node_attrs"], data["charges"]) 
        formal_charge_dipole_contribution = data["positions"] * FQ.unsqueeze(-1)
        formal_charge_dipole = scatter_sum(
            src=formal_charge_dipole_contribution, index=data["batch"].unsqueeze(-1), dim=0, dim_size=num_graphs
        )

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        for interaction, product, readout, charge_map in zip(
            self.interactions, self.products, self.readouts, self.lr_source_maps
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

            # charges
            multipoles_contr, edge_fluxes_contr = charge_map(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                edge_vectors=vectors,
                batch=data["batch"],
                num_graphs=num_graphs
            )  # [n_node, 1, (source_l+1)**2]

            edge_fluxes += edge_fluxes_contr
            charge_density += multipoles_contr.squeeze(-2)

        abs_change = torch.mean(torch.abs(charge_density), dim=-1)
        summed_abs_change = scatter_mean(
            src=abs_change, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        #print("charge density:", summed_abs_change)

        # training and evaluation are different
        if num_scf_steps==0:
            charges_history = [charge_density.clone().detach()]

            field_feats = self.electric_potential_descriptor(
                k_vectors=k_vectors,
                k_vectors_normed_squared=kv_norms_squared,
                k_vectors_mask=kv_mask,
                source_feats=data["density_coefficients"].unsqueeze(-2),
                node_positions=data["positions"],
                batch=data["batch"],
                volumes=data["volume"],
                pbc=data["pbc"].view(-1,3)
            )
            field_feats += self.external_field_contribution(data["batch"], external_field) # for now, this should be zero

            multipoles_contr, edge_fluxes_contr = self.field_dependent_charges_map(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                field_feats=field_feats,
            )
            edge_fluxes += edge_fluxes_contr
            charge_density += multipoles_contr.squeeze(-2)
            charges_history.append(charge_density.clone().detach())

            abs_change = torch.mean(torch.abs(charges_history[-1] - charges_history[-2]), dim=-1)
            summed_abs_change = scatter_mean(
                src=abs_change, index=data["batch"], dim=-1, dim_size=num_graphs
            )
            #print("training mean abs change:", summed_abs_change)

            #print("charge density:", torch.mean(torch.abs(multipoles_contr)))
            #print("charge edge_fluxes:", torch.mean(torch.abs(edge_fluxes)))
            
            charges_history = torch.stack(charges_history, dim=-1)

            fermi_level = torch.zeros((num_graphs, 4), dtype=data["positions"].dtype, device=data["positions"].device)
        else:
            # simpler than polarizable
            # first step happens with zero internal field (+ external field)
            external_field_part = self.external_field_contribution(data["batch"], external_field)
            fermi_level = external_field.clone().detach()

            charges_history = [charge_density.clone().detach()]
            field_independent_charge_density = charge_density.clone()
            field_independent_edge_fluxes = edge_fluxes.clone()

            multipoles_contr_m1 = torch.zeros_like(charge_density).unsqueeze(-2)
            edge_fluxes_contr_m1 = torch.zeros_like(edge_fluxes)

            for step_i in range(num_scf_steps-1):
                field_feats = self.electric_potential_descriptor(
                    k_vectors=k_vectors,
                    k_vectors_normed_squared=kv_norms_squared,
                    k_vectors_mask=kv_mask,
                    source_feats=charge_density.unsqueeze(-2),
                    node_positions=data["positions"],
                    batch=data["batch"],
                    volumes=data["volume"],
                    pbc=data["pbc"].view(-1,3)
                )
                # add external field and fermi level
                field_feats += external_field_part

                multipoles_contr, edge_fluxes_contr = self.field_dependent_charges_map(
                    node_attrs=data["node_attrs"],
                    node_feats=node_feats,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=data["edge_index"],
                    field_feats=field_feats,
                )

                # mixing - for this model it seems to be required
                multipoles_contr = mixing_parameter * multipoles_contr + (1-mixing_parameter) * multipoles_contr_m1
                edge_fluxes_contr = mixing_parameter * edge_fluxes_contr + (1-mixing_parameter) * edge_fluxes_contr_m1
                
                charge_density = field_independent_charge_density + multipoles_contr.squeeze(-2)
                edge_fluxes = field_independent_edge_fluxes + edge_fluxes_contr

                multipoles_contr_m1 = multipoles_contr
                edge_fluxes_contr_m1 = edge_fluxes_contr

                # scf logging
                charges_history.append(charge_density.clone().detach())
                abs_change = torch.mean(torch.abs(charges_history[-1] - charges_history[-2]), dim=-1)
                summed_abs_change = scatter_mean(
                    src=abs_change, index=data["batch"], dim=-1, dim_size=num_graphs
                )
                #print("scf step:", step_i, "mean abs change:", summed_abs_change)

            charges_history = torch.stack(charges_history, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        polarization = compute_polarization(
            charge_density,
            edge_fluxes.squeeze(-1), 
            vectors,
            data["edge_index"],
            data["batch"],
            num_graphs
        )

        polarization += formal_charge_dipole

        # electrostatic energy
        electro_energy = self.coulomb_energy(
            charge_density,
            data["positions"],
            data["batch"],
            data["cell"].view(-1, 3, 3),
            data["rcell"].view(-1, 3, 3),
            data["volume"],
            data["pbc"].view(-1,3),
            num_graphs,
        )
        total_energy += electro_energy

        # Outputs
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "charges_history": charges_history,
            "displacement": displacement,
            "density_coefficients": charge_density,
            "dipole": polarization
        }