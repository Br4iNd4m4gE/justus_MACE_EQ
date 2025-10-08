from typing import Any, Callable, Dict, List, Optional, Type
import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum, scatter_mean
from mace.modules.utils import (
    get_symmetric_displacement,
    get_edge_vectors_and_lengths,
    get_outputs,
)

def permute_to_e3nn_convention(x):
    """ see the note on real harmonics values in https://docs.e3nn.org/en/latest/api/o3/o3_sh.html """
    return x[..., torch.LongTensor([1, 2, 0])]

from mace.modules import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
)

from graph_longrange.kspace import (
    compute_k_vectors
)
from graph_longrange.gto_electrostatics import (
    gto_basis_kspace_cutoff,
    GTOExternalFieldBlock,
    DisplacedGTOExternalFieldBlock,
    KSpaceDirectElectrostaticEnergyBlock,
    PBCAgnosticDirectElectrostaticEnergyBlock
)
from .field_blocks import (
    PureElectrostaticsLRFeatureBlock,
    EnvironmentDependentSourceBlock,
    LinearInFieldChargesBlock,
    MLPNonLinearFieldChargesBlock,
    field_update_blocks,
    field_readout_blocks,
    QuadraticFieldEnergyReadout,
    StrictQuadraticFieldEnergyReadout,
    PBCAgnosticElectrostaticFeatureBlock
)
from .utils import compute_total_charge_dipole


@compile_mode("script")
class Polarizable(torch.nn.Module):
    """ fit charges from local and non-local features """
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
        gate: Optional[Callable],
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        kspace_cutoff_factor: float = 1.5,
        atomic_multipoles_max_l: int = 0, 
        atomic_multipoles_smearing_width: float = 1.0,
        field_feature_widths: List[float] = [1.0],
        include_electrostatic_self_interaction: bool = False,
        add_local_electron_energy: bool = False,
        field_dependence_type: str = "local_linear",
        final_field_readout_type: str = "StrictQuadraticFieldEnergyReadout",
        quadrupole_feature_corrections: bool = False,
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
        self.electric_potential_descriptor = PBCAgnosticElectrostaticFeatureBlock(
            density_max_l=atomic_multipoles_max_l, 
            density_smearing_width=atomic_multipoles_smearing_width, 
            projection_max_l=atomic_multipoles_max_l, 
            projection_smearing_widths=field_feature_widths,
            kspace_cutoff=kspace_cutoff,
            quadrupole_feature_corrections=quadrupole_feature_corrections
        )

        # field dependant charges. TP between node features and local field features. 
        self.charges_irreps = o3.Irreps.spherical_harmonics(atomic_multipoles_max_l)
        lr_sh_irreps = o3.Irreps.spherical_harmonics(atomic_multipoles_max_l)
        self.potential_irreps = (lr_sh_irreps * len(field_feature_widths)).sort()[0].simplify()

        # density
        # the charge is computed after the product
        self.lr_source_maps = torch.nn.ModuleList([EnvironmentDependentSourceBlock(
            irreps_in=hidden_irreps, 
            max_l=atomic_multipoles_max_l
        )])

        for i in range(num_interactions - 1):
            # get rid of the hidden_irreps condition because we need it for charges. 
            hidden_irreps_out = hidden_irreps

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

            # generate charges
            self.lr_source_maps.append(EnvironmentDependentSourceBlock(
                irreps_in=hidden_irreps, 
                max_l=atomic_multipoles_max_l,
            ))

        lr_source_cls = field_update_blocks[field_dependence_type]
        self.field_dependent_charges_map = lr_source_cls(
            node_feat_irreps=hidden_irreps,
            potential_irreps=self.potential_irreps,
            charges_irreps=self.charges_irreps
        )

        self.external_field_contribution = DisplacedGTOExternalFieldBlock(
            atomic_multipoles_max_l,
            field_feature_widths,
            "receiver"
        )
        self.add_local_electron_energy = add_local_electron_energy
        field_readout_cls = field_readout_blocks[final_field_readout_type]
        self.local_electron_energy = field_readout_cls(
            node_feat_irreps=hidden_irreps,
            charges_irreps=self.charges_irreps,
            potential_irreps=self.potential_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
        )
        self.coulomb_energy = PBCAgnosticDirectElectrostaticEnergyBlock(
            density_max_l=atomic_multipoles_max_l,
            density_smearing_width=atomic_multipoles_smearing_width,
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=include_electrostatic_self_interaction
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
        constant_charge: bool = True, # this can be False for constant fermi level, or True for constant charge
        mixing_parameter: float = 1.0,
        scf_training_options: Dict[str, Any] = {},
        use_pbc_evaluator: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert not any((
            compute_virials,
            compute_stress,
            compute_displacement
        ))

        # Setup
        if training:
            for p in self.parameters():
                p.requires_grad = True
            
            # optionally freeze all weights except for those in local_electron_energy
            if "refine" in scf_training_options and scf_training_options["refine"]:
                for name, param in self.named_parameters():
                    if "local_electron_energy" not in name:
                        param.requires_grad = False  
        else:
            for p in self.parameters():
                p.requires_grad = False



        # Setup
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

        external_field = torch.hstack((data['fermi_level'].unsqueeze(-1), data['external_field'])) # [n_graphs, 4]
        
        # density
        charge_density = torch.zeros(
            (data["batch"].size(-1), self.charges_irreps.dim), 
            device=data["batch"].device, dtype=torch.get_default_dtype()
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
        #edge_feats = self.radial_embedding(lengths)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # kgrid - note that stress doesn't work yet sicne you need the der. w.r.t. rcell
        k_vectors, kv_norms_squared, kv_mask = compute_k_vectors(
            self.kspace_cutoff, data["cell"].view(-1,3,3), data["rcell"].view(-1,3,3)
        )

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        for (interaction, product, readout, lr_source_map) in zip(
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

            # readouts
            charge_sources = lr_source_map(
                node_feats
            ) # [n_node, 1, (source_l+1)**2]

            charge_density += charge_sources.squeeze(-2)

        # scf bit. training requires the true potential. 
        field_independent_charge_density = charge_density.clone()
        if num_scf_steps==0:
            field_feats = self.electric_potential_descriptor(
                k_vectors=k_vectors,
                k_vectors_normed_squared=kv_norms_squared,
                k_vectors_mask=kv_mask,
                source_feats=data["density_coefficients"].unsqueeze(-2),
                node_positions=data["positions"],
                batch=data["batch"],
                volumes=data["volume"],
                pbc=data["pbc"].view(-1,3),
                use_pbc_evaluator=use_pbc_evaluator
            )
            field_feats += self.external_field_contribution(data["batch"], data["positions"], external_field) # for now, this should be zero

            charge_sources = self.field_dependent_charges_map(
                node_feats,
                field_feats
            )
            charge_density += charge_sources # [n_nodes, ]

            charges_history = [charge_density.detach()]
            charges_history = torch.stack(charges_history, dim=-1)
            fermi_level = torch.zeros((num_graphs, 4), dtype=data["positions"].dtype, device=data["positions"].device)

            field_feats_out = self.electric_potential_descriptor(
                k_vectors=k_vectors,
                k_vectors_normed_squared=kv_norms_squared,
                k_vectors_mask=kv_mask,
                source_feats=charge_density.unsqueeze(-2),
                node_positions=data["positions"],
                batch=data["batch"],
                volumes=data["volume"],
                pbc=data["pbc"].view(-1,3),
                use_pbc_evaluator=use_pbc_evaluator
            )
        else:
            # we have two conditions for constant field and constant charge
            if not constant_charge:
                # first step happens with zero internal field (+ external field)
                fixed_external_field = self.external_field_contribution(data["batch"], data['positions'], external_field)
                charges_history = [charge_density.detach()]
                field_independent_charge_density = charge_density.clone()

                for step_i in range(num_scf_steps-1):
                    # get internal field
                    field_feats_out = self.electric_potential_descriptor(
                        k_vectors=k_vectors,
                        k_vectors_normed_squared=kv_norms_squared,
                        k_vectors_mask=kv_mask,
                        source_feats=charge_density.unsqueeze(-2),
                        node_positions=data["positions"],
                        batch=data["batch"],
                        volumes=data["volume"],
                        pbc=data["pbc"].view(-1,3),
                        use_pbc_evaluator=use_pbc_evaluator
                    )
                    if step_i==0:
                        field_feats = field_feats_out.clone()

                    # add external field and fermi level
                    field_feats_out += fixed_external_field

                    # compute charges
                    field_dep_contribution = self.field_dependent_charges_map(
                        node_feats,
                        field_feats_out
                    ).squeeze(-2)
                    
                    # update with mixing
                    Delta_n = field_independent_charge_density + field_dep_contribution - charge_density
                    charge_density = charge_density + Delta_n * mixing_parameter

                    # logging
                    charges_history.append(charge_density.detach().clone())
                    abs_change = torch.mean(torch.abs(charges_history[-1] - charges_history[-2]), dim=-1)
                    summed_abs_change = scatter_mean(
                        src=abs_change, index=data["batch"], dim=-1, dim_size=num_graphs
                    )

                # charge_density = charge_density.detach()
                charges_history = torch.stack(charges_history, dim=-1)

                total_charge, total_dipole = compute_total_charge_dipole(
                    charge_density,
                    data["positions"],
                    data["batch"],
                    num_graphs
                )

                field_feats = self.electric_potential_descriptor(
                    k_vectors=k_vectors,
                    k_vectors_normed_squared=kv_norms_squared,
                    k_vectors_mask=kv_mask,
                    source_feats=charge_density.unsqueeze(-2),
                    node_positions=data["positions"],
                    batch=data["batch"],
                    volumes=data["volume"],
                    pbc=data["pbc"].view(-1,3),
                    use_pbc_evaluator=use_pbc_evaluator
                )

                fermi_level = external_field

            else: # total charge condition
                target_charge = data["total_charge"]
                # try to vary fermi level by newtons method
                # get derivative of charge w.r.t fermi level (v = dQ/dmu), then set mu=DQ/v
                fixed_external_field = self.external_field_contribution(data["batch"], data['positions'], external_field)
                fermi_level = torch.zeros((num_graphs, 4), dtype=torch.get_default_dtype(), device=data["positions"].device)

                charges_history = [charge_density.detach()]
                field_independent_charge_density = charge_density.clone()

                # below, the squeezing is done to keep everything like 'charge density'
                field_dep_contribution_m1 = torch.zeros_like(charge_density)

                for step_i in range(num_scf_steps-1):
                    fermi_level.requires_grad_(True)

                    # get internal field
                    field_feats_out = self.electric_potential_descriptor(
                        k_vectors=k_vectors,
                        k_vectors_normed_squared=kv_norms_squared,
                        k_vectors_mask=kv_mask,
                        source_feats=charge_density.unsqueeze(-2),
                        node_positions=data["positions"],
                        batch=data["batch"],
                        volumes=data["volume"],
                        pbc=data["pbc"].view(-1,3),
                        use_pbc_evaluator=use_pbc_evaluator
                    )
                    if step_i==0:
                        field_feats = field_feats_out.clone()
                    # add external field and fermi level
                    field_feats_out += fixed_external_field + self.external_field_contribution(data["batch"], data['positions'], fermi_level)

                    # compute charges
                    field_dep_contribution = self.field_dependent_charges_map(
                        node_feats,
                        field_feats_out
                    ).squeeze(-2)

                    pred_total_charge = scatter_sum(
                        src=field_independent_charge_density[:,0] + field_dep_contribution[:,0], index=data["batch"], dim=-1, dim_size=num_graphs
                    ).detach()
                    
                    # update with mixing
                    Delta_n = field_independent_charge_density + field_dep_contribution - charge_density
                    charge_density = charge_density + Delta_n * mixing_parameter

                    # logging
                    charges_history.append(charge_density.detach().clone())
                    abs_change = torch.mean(torch.abs(charges_history[-1] - charges_history[-2]), dim=-1)
                    summed_abs_change = scatter_mean(
                        src=abs_change, index=data["batch"], dim=-1, dim_size=num_graphs
                    )

                    total_charge = scatter_sum(
                        src=charge_density[:,0], index=data["batch"], dim=-1, dim_size=num_graphs
                    )
                    # gradient
                    grad_outputs = [torch.ones_like(total_charge)]
                    gradient = torch.autograd.grad(
                        outputs=[total_charge],  # [n_graphs, ]
                        inputs=[fermi_level],  # [n_graphs, 4]
                        grad_outputs=grad_outputs,
                        retain_graph=compute_force,  
                        allow_unused=True,  # For complete dissociation turn to true
                    )[0][:,0]

                    fermi_level = fermi_level.detach()
                    if (not training) and (not compute_force):
                        fermi_level = fermi_level.detach()
                        total_charge = total_charge.detach()
                        field_feats = field_feats.detach()
                        charge_density = charge_density.detach()
                        torch.cuda.empty_cache()
                    
                    # update fermi level
                    small_gradients = torch.abs(gradient) < 1e-6

                    delta_Q = target_charge - pred_total_charge # * (1 - mixing_parameter) - mixing_parameter * pred_total_charge
                    delta_Q[small_gradients] = 1.0

                    delta_ef = torch.divide(delta_Q, gradient / mixing_parameter)
                    delta_ef[small_gradients] = -1.0
                    if step_i >= 0:
                        fermi_level[...,0] += delta_ef.clamp(-1.,1.)
                
                # charge_density = charge_density.detach()
                charges_history = torch.stack(charges_history, dim=-1)

                total_charge, total_dipole = compute_total_charge_dipole(
                    charge_density,
                    data["positions"],
                    data["batch"],
                    num_graphs
                )

                field_feats_out = self.electric_potential_descriptor(
                    k_vectors=k_vectors,
                    k_vectors_normed_squared=kv_norms_squared,
                    k_vectors_mask=kv_mask,
                    source_feats=charge_density.unsqueeze(-2),
                    node_positions=data["positions"],
                    batch=data["batch"],
                    volumes=data["volume"],
                    pbc=data["pbc"].view(-1,3),
                    use_pbc_evaluator=use_pbc_evaluator
                )

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        local_q_e = self.local_electron_energy(
            node_feats=node_feats,
            charges_0=field_independent_charge_density,
            charges_induced=charge_density,
            edge_feats=edge_feats,
            edge_attrs=edge_attrs,
            field_feats=field_feats_out,
            edge_index=data["edge_index"],
            batch=data["batch"],
        )
        le_total = scatter_sum(
            src=local_q_e, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        if self.add_local_electron_energy:  
            total_energy += le_total
        else:
            # since this is returned
            le_total = torch.zeros_like(le_total)

        total_charge, total_dipole = compute_total_charge_dipole(
            charge_density,
            data["positions"],
            data["batch"],
            num_graphs
        )

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
            use_pbc_evaluator=use_pbc_evaluator,
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
            "energy" : total_energy,
            "forces" : forces,
            "density_coefficients": charge_density,
            "charges_history": charges_history,
            "fermi_level": fermi_level.detach()[:,0],
            "dipole": total_dipole,
            "electrostatic_energy": electro_energy,
            "electron_energy": le_total,
        }