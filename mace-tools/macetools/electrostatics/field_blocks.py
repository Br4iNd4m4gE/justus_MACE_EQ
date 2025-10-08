from typing import Callable, Optional, Tuple, Union, List
import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode
from mace.tools.scatter import scatter_sum
from mace.modules.irreps_tools import tp_out_irreps_with_instructions
import numpy as np

from graph_longrange.gto_electrostatics import (
    GTOFourierSeriesCoeficientsBlock,
    GTOChargeDensityFourierSeriesBlock,
    GTOLocalOrbitalProjectionBlock,
    KspaceCoulombOperatorBlock,
    GTOSelfInteractionBlock,
    MonopoleDipoleFieldBlock,
    DisplacedGTOExternalFieldBlock,
    CorrectivePotentialBlock,
    GTOInternalFieldtoFeaturesBlock
)
from graph_longrange.realspace_electrostatics import RealSpaceLRFeatureBlock

from .utils import compute_polarization


@compile_mode("script")
class PerAtomFormalChargesBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node_attr: torch.Tensor, node_charges: torch.Tensor):
        return node_charges


@compile_mode("script")
class PerSpeciesFormalChargesBlock(torch.nn.Module):
    formal_charges: torch.Tensor

    def __init__(
        self,
        formal_charges: Union[np.ndarray, torch.Tensor],
    ):
        super().__init__()
        assert len(formal_charges.shape) == 1
        self.register_buffer(
            "formal_charges",
            torch.tensor(formal_charges, dtype=torch.get_default_dtype()),
        )  # [n_elements, ]

    def forward(
        self, node_attr: torch.Tensor, node_charges: torch.Tensor
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(node_attr, self.formal_charges)

    def __repr__(self):
        formatted_energies = ", ".join([f"{x:.4f}" for x in self.formal_charges])
        return f"{self.__class__.__name__}(charges=[{formatted_energies}])"
    

@compile_mode("script")
class EnvironmentDependentSourceBlock(torch.nn.Module):
    def __init__(
        self, 
        irreps_in: o3.Irreps, 
        max_l: int,
        zero_charges: bool = False
    ):
        super().__init__()
        irreps_out = o3.Irreps.spherical_harmonics(max_l)
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)
        self.zero_charges = zero_charges

    def forward(
        self,
        node_feats: torch.Tensor # [n_node, hidden_irreps.dim]
    ) -> torch.Tensor:
        mpoles = self.linear(node_feats) # [n_node, (max_l+1)**2]
        if self.zero_charges:
            zeroed_mpoles = torch.zeros_like(mpoles)
            zeroed_mpoles[:,1:] = mpoles[:,1:]
        else:
            zeroed_mpoles = mpoles
        return zeroed_mpoles.unsqueeze(-2) # [n_node, 1, (max_l+1)**2]


@compile_mode("script")
class NoFieldSymmetricPredictionSourceBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        max_l: int,
    ):
        super().__init__()

        self.multipole_block = EnvironmentDependentSourceBlock(
            irreps_in=node_feats_irreps,
            max_l=max_l,
            zero_charges=True
        )

        self.linear1 = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.linear2 = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps,
            edge_attrs_irreps,
            target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            node_feats_irreps,
            edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        # Convolution weights
        input_dim = edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # Linear
        irreps_mid = irreps_mid.simplify()
        irreps_out = o3.Irreps("1x0e")  # map straight to a scalar
        self.linear = o3.Linear(
            irreps_mid, irreps_out, internal_weights=True, shared_weights=True
        )

    def forward(
        self,
        node_attrs: torch.Tensor,  # [n_node, num_el]
        node_feats: torch.Tensor,  # [n_node, hidden_irreps.dim]
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        tp_weights = self.conv_tp_weights(edge_feats)
        bond_feats = (
            self.linear1(node_feats)[sender] + self.linear2(node_feats)[receiver]
        )
        mji = self.conv_tp(bond_feats, edge_attrs, tp_weights)  # [n_edges, irreps]

        p_ji = self.linear(mji).squeeze(-1) / 40.0

        # sum over edges
        charges = scatter_sum(
            src=p_ji, index=receiver, dim=0, dim_size=num_nodes
        ) - scatter_sum(src=p_ji, index=sender, dim=0, dim_size=num_nodes)

        multipoles = self.multipole_block(node_feats)
        multipoles[:,0,0] = charges

        return multipoles.squeeze(-2), p_ji.unsqueeze(-1) # [n_node, 1, (max_l+1)**2]


@compile_mode("script")
class LinearFieldSymmetricSourceBlock(torch.nn.Module):
    """
    works very simililarly to NoFieldSymmetricPredictionSourceBlock, expect that the final output is not 
    the charge, but instead the gradients with respect to field components.
    """

    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        charges_irreps: o3.Irreps,
        field_feats_irreps: o3.Irreps,
        target_irreps,
    ):
        super().__init__()

        # multipoles bit
        self.field_dependent_mpoles = LinearInFieldChargesBlock(
            node_feat_irreps=node_feats_irreps,
            potential_irreps=field_feats_irreps,
            charges_irreps=charges_irreps,
            zero_charges=True
        )

        self.linear1 = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.linear2 = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps,
            edge_attrs_irreps,
            target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            node_feats_irreps,
            edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        # Convolution weights
        input_dim = edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        # Linear to gradients
        irreps_mid = irreps_mid.simplify()
        self.linear_v_sender = o3.Linear(
            irreps_mid, o3.Irreps(field_feats_irreps), internal_weights=True, shared_weights=True
        )
        self.linear_v_receiver = o3.Linear(
            irreps_mid, o3.Irreps(field_feats_irreps), internal_weights=True, shared_weights=True
        )
        # final dot products
        self.final_tp_sender = o3.FullyConnectedTensorProduct(
            irreps_in1=field_feats_irreps,
            irreps_in2=field_feats_irreps,
            irreps_out=o3.Irreps("1x0e")
        )
        self.final_tp_receiver = o3.FullyConnectedTensorProduct(
            irreps_in1=field_feats_irreps,
            irreps_in2=field_feats_irreps,
            irreps_out=o3.Irreps("1x0e")
        )

    def forward(
        self,
        node_attrs: torch.Tensor,  # [n_node, num_el]
        node_feats: torch.Tensor,  # [n_node, hidden_irreps.dim]
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        tp_weights = self.conv_tp_weights(edge_feats)
        bond_feats = (
            self.linear1(node_feats)[sender] + self.linear2(node_feats)[receiver]
        )
        mji = self.conv_tp(bond_feats, edge_attrs, tp_weights)  # [n_edges, irreps]
        A = self.linear_v_sender(mji)
        B = self.linear_v_receiver(mji)

        p_ji = self.final_tp_sender(A, field_feats[sender]) + self.final_tp_receiver(B, field_feats[receiver])
        p_ji = 0.1 * p_ji.squeeze(-1)

        # sum over edges
        charges = scatter_sum(
            src=p_ji, index=receiver, dim=0, dim_size=num_nodes
        ) - scatter_sum(src=p_ji, index=sender, dim=0, dim_size=num_nodes)

        multipoles = self.field_dependent_mpoles(node_feats, field_feats)
        multipoles = multipoles.unsqueeze(-2)
        multipoles[:,0,0] = charges

        return multipoles, p_ji.unsqueeze(-1) # [n_node, 1, (max_l+1)**2]


@compile_mode("script")
class PBCAgnosticElectrostaticFeatureBlock(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        projection_max_l: int,
        projection_smearing_widths: List[float],
        kspace_cutoff: float,        
        include_self_interaction=False,
        integral_normalization="receiver",
        quadrupole_feature_corrections=False
    ):
        super().__init__()

        self.pbc_features = PureElectrostaticsLRFeatureBlock(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            projection_max_l=projection_max_l,
            projection_smearing_widths=projection_smearing_widths,
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=include_self_interaction,
            integral_normalization=integral_normalization,
            quadrupole_feature_corrections=quadrupole_feature_corrections
        )

        self.realspace_features = RealSpaceLRFeatureBlock(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            projection_max_l=projection_max_l,
            projection_smearing_widths=projection_smearing_widths,
            include_self_interaction=include_self_interaction,
            integral_normalization=integral_normalization,
        )

    def forward(
        self,
        k_vectors: torch.Tensor,
        k_vectors_normed_squared: torch.Tensor,
        k_vectors_mask: torch.Tensor,
        source_feats: torch.Tensor, # [n_nodes, 1, (max_l_s+1)**2]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
        pbc: torch.Tensor,
        use_pbc_evaluator: bool,
    ) -> torch.Tensor:
        if pbc[0,0] or use_pbc_evaluator:
            return self.pbc_features(
                k_vectors,
                k_vectors_normed_squared,
                k_vectors_mask,
                source_feats,
                node_positions,
                batch,
                volumes,
                pbc
            )
        else:
            return self.realspace_features(
                source_feats,
                node_positions,
                batch
            )

@compile_mode("script")
class PureElectrostaticsLRFeatureBlock(torch.nn.Module):
    """ this takes a set of source weights for GTO basis of charge density, and computes the field due to the density
        returns the projections of this field onto a second local basis of GTOs. 
        this is the field method block.  """
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        projection_max_l: int,
        projection_smearing_widths: List[float],
        kspace_cutoff: float,        
        include_self_interaction=False,
        integral_normalization="receiver",
        quadrupole_feature_corrections=False
    ):
        super().__init__()
        self.include_self_interaction = include_self_interaction

        # density
        self.density_gto_fs_block = GTOFourierSeriesCoeficientsBlock(
            sigmas=[density_smearing_width],
            max_l=density_max_l,
            kspace_cutoff=kspace_cutoff,
            normalize='multipoles'
        )
        self.density_block = GTOChargeDensityFourierSeriesBlock()

        # convolve
        self.field_conv_operator = KspaceCoulombOperatorBlock()

        # project
        self.project_gto_fs_block = GTOFourierSeriesCoeficientsBlock(
            sigmas=projection_smearing_widths,
            max_l=projection_max_l,
            kspace_cutoff=kspace_cutoff,
            normalize=integral_normalization
        )
        self.projector = GTOLocalOrbitalProjectionBlock()

        num_radial_channels = len(projection_smearing_widths)
        indices = []
        for l in range(projection_max_l+1):
            for c in range(num_radial_channels):
                offset = c*(projection_max_l+1)**2
                indices += range(l**2+offset, (l+1)**2 + offset)
        
        self.register_buffer(
            "indices", torch.tensor(indices, dtype=torch.int64)
        )

        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            projection_max_l, 
            projection_smearing_widths,
            "multipoles",
            integral_normalization
        )

        self.non_periodic_correction_terms = NPCCorrectsFeatureBlock(
            density_max_l,
            projection_max_l,
            projection_smearing_widths,
            integral_normalization,
            quadrupole_feature_corrections=quadrupole_feature_corrections,
        )

    def forward(
        self,
        k_vectors: torch.Tensor,
        k_vectors_normed_squared: torch.Tensor,
        k_vectors_mask: torch.Tensor,
        source_feats: torch.Tensor, # [n_nodes, 1, (max_l_s+1)**2]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
        pbc: torch.Tensor
    ) -> torch.Tensor:

        basis_fs = self.density_gto_fs_block(
            k_vectors, 
            k_vectors_normed_squared, 
            k_vectors_mask
        ) # [n_graph, max_n_k, 1, (max_l_s+1)**2, 2]
        #print(f"basis_fs.shape : {basis_fs.shape}")

        density = self.density_block(
            source_feats,
            node_positions,
            k_vectors,
            basis_fs,
            volumes,
            batch
        ) # [n_graph, max_n_k, 2]
        #print(f"density.shape : {density.shape}")

        potential = self.field_conv_operator(
            density,
            k_vectors_normed_squared,
            k_vectors_mask
        ) # [n_graph, max_n_k, 2]
        #print(f"potential.shape : {potential.shape}")

        basis_fs = self.project_gto_fs_block(
            k_vectors, 
            k_vectors_normed_squared, 
            k_vectors_mask
        ) # [n_graph, max_n_k, n_receive_radial, (max_l_r+1)**2, 2]
        #print(f"basis_fs_2.shape : {basis_fs.shape}")

        projections = self.projector(
            k_vectors,
            node_positions,
            potential,
            batch,
            k_vectors_mask,
            basis_fs
        ) # [n_nodes, n_sigma, (max_l+1)**2]
        #print(f"projections.shape : {projections.shape}")

        reshaped = projections.flatten(start_dim=-2)[:,self.indices] 
        if not self.include_self_interaction:
            reshaped -= self.self_interaction(source_feats.squeeze(-2))

        # need to do pbc checks
        is_pbc = torch.index_select(torch.all(pbc, dim=1), -1,  batch)
        correction_terms = self.non_periodic_correction_terms(
            source_feats,
            node_positions,
            batch,
            volumes
        )
        corrections = torch.where(
            is_pbc.unsqueeze(-1), 
            torch.zeros_like(correction_terms),
            correction_terms
        )
        reshaped += corrections

        return reshaped


@compile_mode("script")
class NPCCorrectsFeatureBlock(torch.nn.Module):
    def __init__(
            self,
            density_max_l: int,
            projection_max_l: int,
            projection_smearing_widths: List[float],
            integral_normalization="receiver",
            quadrupole_feature_corrections=False
        ):
        super().__init__()
        self.self_field = CorrectivePotentialBlock(
            density_max_l=density_max_l,
            quadrupole_feature_corrections=quadrupole_feature_corrections
        )
        self.displaced_interactions = GTOInternalFieldtoFeaturesBlock(
            l_receive=projection_max_l,
            sigmas_receive=projection_smearing_widths,
            normalize_receive=integral_normalization
        )

    def forward(
        self,
        source_feats: torch.Tensor, # [n_nodes, 1, (max_l_s+1)**2]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor
    ) -> torch.Tensor:
        node_fields = self.self_field(
            charge_coefficients=source_feats.squeeze(-2),
            positions=node_positions,
            volumes=volumes,
            batch=batch
        ) # [V, Ex, Ey, Ez]
        projections = self.displaced_interactions(
            batch=batch,
            positions=node_positions,
            node_fields=node_fields
        )
        return projections


@compile_mode("script")
class LinearInFieldChargesBlock(torch.nn.Module):
    def __init__(
        self,
        node_feat_irreps: o3.Irreps,
        potential_irreps: o3.Irreps,
        charges_irreps: o3.Irreps,
        zero_charges=False,
    ):
        super().__init__()

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=node_feat_irreps,
            irreps_in2=potential_irreps,
            irreps_out=charges_irreps
        )

        self.zero_charges = zero_charges

    def forward(self, node_feat, pot_feat):
        mpoles = 0.01 * self.tp(node_feat, pot_feat)
        if self.zero_charges:
            zeroed_mpoles = torch.zeros_like(mpoles)
            zeroed_mpoles[:,1:] = mpoles[:,1:]
        else:
            zeroed_mpoles = mpoles
        return zeroed_mpoles # [n_node, (l+1)^2]


@compile_mode("script")
class BiasedLinearInFieldChargesBlock(torch.nn.Module):
    def __init__(
        self,
        node_feat_irreps: o3.Irreps,
        potential_irreps: o3.Irreps,
        charges_irreps: o3.Irreps,
        zero_charges=False,
    ):
        super().__init__()

        self.linear1 = o3.Linear(
            potential_irreps, potential_irreps, biases=True
        )

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=node_feat_irreps,
            irreps_in2=potential_irreps,
            irreps_out=charges_irreps
        )

        self.zero_charges = zero_charges

    def forward(self, node_feat, pot_feats):
        pot_feats_in = self.linear1(0.01 * pot_feats)
        mpoles = 0.01 * self.tp(node_feat, pot_feats_in)
        if self.zero_charges:
            zeroed_mpoles = torch.zeros_like(mpoles)
            zeroed_mpoles[:,1:] = mpoles[:,1:]
        else:
            zeroed_mpoles = mpoles
        return zeroed_mpoles # [n_node, (l+1)^2]


@compile_mode("script")
class ChargesReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        node_feat_irreps: o3.Irreps,
        charges_irreps: o3.Irreps
    ):
        super().__init__()
        self.linear_1 = o3.Linear(node_feat_irreps, node_feat_irreps)
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=node_feat_irreps,
            irreps_in2=charges_irreps,
            irreps_out=o3.Irreps("1x0e")
        )

    def forward(
        self, 
        node_feats: torch.Tensor, 
        charges: torch.Tensor
    ):
        new_feats = self.linear_1(node_feats)
        energy = self.tp(new_feats, charges)
        return energy.squeeze(-1)
        



@compile_mode("script")
class MLPNonLinearFieldChargesBlock(torch.nn.Module):
    def __init__(
        self,
        node_feat_irreps: o3.Irreps,
        potential_irreps: o3.Irreps,
        charges_irreps: o3.Irreps,
    ):
        super().__init__()
        self.norm = o3.Norm(potential_irreps, squared=False)
        # linear the node features first
        self.linear_up = o3.Linear(
            node_feat_irreps,
            node_feat_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        # potential_sh_irreps = o3.Irreps.spherical_harmonics(potential_irreps.lmax)
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feat_irreps, potential_irreps, charges_irreps
        )
        self.conv_tp = o3.TensorProduct( # we map from Vlm, hklm to Hklm'
            node_feat_irreps,
            potential_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        # Convolution weights
        self.input_dim = (potential_irreps.lmax + 1) * potential_irreps.count(o3.Irrep(0, 1))
        self.conv_tp_weights = nn.FullyConnectedNet(
            [self.input_dim] + 3 * [16] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu
        )
        # linear down
        self.linear_out = o3.Linear(
            irreps_mid, charges_irreps, internal_weights=True, shared_weights=True
        )

    def forward(self, node_feats, pot_feats):
        # needed
        pot_feats = 1.0 * pot_feats
        node_feats = self.linear_up(node_feats)
        potential_norms = self.norm(pot_feats)

        reshaped_norms = potential_norms.view(-1, self.input_dim)

        tp_weights = self.conv_tp_weights(reshaped_norms + 1.0*torch.ones_like(reshaped_norms))
        product = self.conv_tp(
            node_feats, pot_feats, tp_weights
        )
        charges = self.linear_out(product)
        return charges


@compile_mode("script")
class NonSymNonLinearFieldChargesBlock(torch.nn.Module):
    def __init__(
        self,
        node_feat_irreps: o3.Irreps,
        potential_irreps: o3.Irreps,
        charges_irreps: o3.Irreps,
    ):
        super().__init__()

        # product irreps is node_feat_irreps but only l=0
        product_irreps = o3.Irreps(f"{node_feat_irreps.count(o3.Irrep(0, 1))}x0e")

        # linear both features first
        self.linear1 = o3.Linear(
            node_feat_irreps,
            node_feat_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.linear2 = o3.Linear(
            potential_irreps,
            node_feat_irreps,
            biases=True
        )
        # Tensor Product for weights
        self.tp_for_weights = o3.FullyConnectedTensorProduct(
            node_feat_irreps,
            node_feat_irreps,
            product_irreps
        )
        self.norm = o3.Norm(product_irreps, squared=True)

        # Tensor product out
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feat_irreps, potential_irreps, charges_irreps
        )
        self.conv_tp = o3.TensorProduct( # we map from Vlm, hklm to Hklm'
            node_feat_irreps,
            potential_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        # Convolution weights
        self.input_dim = product_irreps.count(o3.Irrep(0, 1))
        self.conv_tp_weights = nn.FullyConnectedNet(
            [self.input_dim] + 3 * [16] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu
        )
        # linear down
        self.linear_out = o3.Linear(
            irreps_mid, charges_irreps, internal_weights=True, shared_weights=True
        )

    def forward(self, node_feats, pot_feats):
        # needed
        node_feats = self.linear1(node_feats)
        pot_feats_upper = self.linear2(pot_feats)
        product = self.tp_for_weights(node_feats, pot_feats_upper)
        norms = self.norm(product)

        tp_weights = self.conv_tp_weights(norms)
        product = self.conv_tp(
            node_feats, pot_feats, tp_weights
        )
        charges = self.linear_out(product)
        return charges


class QuadraticFieldEnergyReadout(torch.nn.Module):
    def __init__(
        self, node_feat_irreps: o3.Irreps, 
        charges_irreps: o3.Irreps, 
        potential_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps
    ):
        super().__init__()

        irreps_mid = o3.Irreps("32x0e + 32x1o")

        # mixing of the fixed and induced components
        self.linear_qa = o3.Linear(charges_irreps, irreps_mid, biases=True)
        self.linear_qb = o3.Linear(charges_irreps, irreps_mid, biases=True)
        self.linear_va = o3.Linear(potential_irreps, irreps_mid, biases=True)
        self.linear_va = o3.Linear(potential_irreps, irreps_mid, biases=True)

        # tensor products between charges and fields
        self.qv_tp = o3.FullyConnectedTensorProduct(
            irreps_mid, irreps_mid, node_feat_irreps
        )
        """ irreps_mid = self.qv_tp.irreps_out
        max_l_mid = irreps_mid.lmax
        n_channels = node_feat_irreps.count(o3.Irrep(0, 1)) """

        # construct the weights tensor to be n_channels by max_l_mid
        """ irr = []
        for ll in range(max_l_mid+1):
            if ll%2 == 0:
                parity = 1
            else:
                parity = -1
            irr.append((n_channels, (ll, parity)))
        irreps_U = o3.Irreps(irr)

        # tensor product from node features to weight space
        self.weights_tp = o3.FullyConnectedTensorProduct(
            node_feat_irreps, node_feat_irreps, irreps_U
        )

        # final contration
        self.tp_down = o3.FullyConnectedTensorProduct(
            irreps_mid, irreps_U, o3.Irreps("1x0e")
        
        ) """

        # tp down
        self.tp_down = o3.FullyConnectedTensorProduct(
            node_feat_irreps, node_feat_irreps, o3.Irreps("1x0e")
        )

    def forward(
        self, 
        node_feats, 
        charges_0,
        charges_induced,
        edge_feats,
        edge_attrs,
        field_feats,
        edge_index,
        batch
    ):
        fields = 0.01 * field_feats
        charges = self.linear_qa(charges_induced - charges_0) # + self.linear_qb(charges_induced)
        fields = self.linear_va(fields) # + self.linear_va(field_induced)

        # tensor product between charges and fields
        qv = self.qv_tp(charges, fields)

        # final contraction
        energy = self.tp_down(qv, node_feats)
    
        return energy.squeeze(-1)


class StrictQuadraticFieldEnergyReadout(torch.nn.Module):
    def __init__(
        self, node_feat_irreps: o3.Irreps, 
        charges_irreps: o3.Irreps, 
        potential_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps
    ):
        super().__init__()

        irreps_mid = o3.Irreps("32x0e + 32x1o")

        # mixing of the fixed and induced components
        self.linear_qa = o3.Linear(charges_irreps, irreps_mid, biases=False)
        self.linear_va = o3.Linear(potential_irreps, irreps_mid, biases=False)

        # tensor products between charges and fields
        self.qv_tp = o3.FullyConnectedTensorProduct(
            irreps_mid, irreps_mid, o3.Irreps("32x0e")
        )

        # tp down
        self.tp_down = o3.FullyConnectedTensorProduct(
            o3.Irreps("32x0e"), node_feat_irreps, o3.Irreps("1x0e")
        )

    def forward(
        self, 
        node_feats, 
        charges_0,
        charges_induced,
        edge_feats,
        edge_attrs,
        field_feats,
        edge_index,
        batch
    ):
        fields = 0.01 * field_feats
        charges = self.linear_qa(charges_induced - charges_0)
        fields = self.linear_va(fields)

        # tensor product between charges and fields
        qv = self.qv_tp(charges, fields)

        # final contraction
        energy = self.tp_down(qv, node_feats)
    
        return energy.squeeze(-1)

    
class QuadraticChargesEnergyReadout(torch.nn.Module):
    def __init__(
        self, node_feat_irreps: o3.Irreps, 
        charges_irreps: o3.Irreps, 
        potential_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps
    ):
        super().__init__()

        irreps_mid = o3.Irreps("32x0e + 32x1o")

        # mixing of the fixed and induced components
        self.linear_qa = o3.Linear(charges_irreps, irreps_mid, biases=True)
        self.linear_qb = o3.Linear(charges_irreps, irreps_mid, biases=True)

        # tensor products between charges and fields
        self.qq_tp = o3.FullyConnectedTensorProduct(
            irreps_mid, irreps_mid, irreps_mid
        )

        # tp down
        self.tp_down = o3.FullyConnectedTensorProduct(
            irreps_mid, node_feat_irreps, o3.Irreps("1x0e")
        )

    def forward(
        self, 
        node_feats, 
        charges_0,
        charges_induced,
        edge_feats,
        edge_attrs,
        field_feats,
        edge_index,
        batch
    ):
        charges = self.linear_qa(charges_induced) + self.linear_qb(charges_0)

        # tensor product between charges and fields
        qq = self.qq_tp(charges, charges)

        # final contraction
        energy = self.tp_down(qq, node_feats)
    
        return energy.squeeze(-1)


class LinearChargesEnergyReadout(torch.nn.Module):
    def __init__(
        self, node_feat_irreps: o3.Irreps, 
        charges_irreps: o3.Irreps, 
        potential_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps
    ):
        super().__init__()

        self.tp = o3.FullyConnectedTensorProduct(
            charges_irreps, node_feat_irreps, o3.Irreps("1x0e")
        )

    def forward(
        self, 
        node_feats, 
        charges_0,
        charges_induced,
        edge_feats,
        edge_attrs,
        field_feats,
        edge_index,
        batch
    ):
        energy = self.tp(charges_induced, node_feats)
    
        return energy.squeeze(-1)

class OneBodyInteractionEnergyReadout(torch.nn.Module):
    def __init__(
        self, node_feat_irreps: o3.Irreps, 
        charges_irreps: o3.Irreps, 
        potential_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps
    ):
        super().__init__()

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            charges_irreps,
            edge_attrs_irreps,
            node_feat_irreps,
        )
        self.edge_tp = o3.TensorProduct(
            charges_irreps,
            edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        # Convolution weights
        input_dim = edge_feats_irreps.num_irreps
        self.edge_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.edge_tp.weight_numel],
            torch.nn.functional.silu,
        )
        self.node_tp = o3.FullyConnectedTensorProduct(
            irreps_mid, node_feat_irreps, o3.Irreps("1x0e")
        )


    def forward(
        self, 
        node_feats, 
        charges_0,
        charges_induced,
        edge_feats,
        edge_attrs,
        field_feats,
        edge_index,
        batch
    ):
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        tp_weights = self.edge_tp_weights(edge_feats)
        edge_q_rep = self.edge_tp(charges_induced[sender], edge_attrs, tp_weights) # [..., node_feats_irrpes]
        summed_charge_rep = scatter_sum(
            src=edge_q_rep, index=receiver, dim=0, dim_size=num_nodes
        )
        energy = self.node_tp(summed_charge_rep, node_feats)

        return energy.squeeze(-1)


        

field_update_blocks = {
    "local_linear": LinearInFieldChargesBlock,
    "biased_local_linear": BiasedLinearInFieldChargesBlock,
    "local_mlp": MLPNonLinearFieldChargesBlock,
    "nonsym_local_mlp":NonSymNonLinearFieldChargesBlock
}

field_readout_blocks = {
    "StrictQuadraticFieldEnergyReadout": StrictQuadraticFieldEnergyReadout,
    "QuadraticFieldEnergyReadout": QuadraticFieldEnergyReadout,
    "LinearChargesEnergyReadout": LinearChargesEnergyReadout,
    "OneBodyInteractionEnergyReadout": OneBodyInteractionEnergyReadout,
    "QuadraticChargesEnergyReadout": QuadraticChargesEnergyReadout
}
