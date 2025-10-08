from numpy import cov
import torch

from typing import Dict, Optional

from mace.modules.models import MACE

from e3nn import o3
from e3nn.util.jit import compile_mode

from macetools.electrostatics.qeq_block import (
    ChargeEquilibrationBlock
)
from macetools.electrostatics.utils import compute_coulomb_energy
from mace.tools.scatter import scatter_sum


from mace.modules.blocks import (
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    ScaleShiftBlock,
    LinearNodeEmbeddingBlock,
)
from mace.modules.utils import (
    compute_fixed_charge_dipole,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)

from graph_longrange.gto_electrostatics import (
    PBCAgnosticDirectElectrostaticEnergyBlock,
    gto_basis_kspace_cutoff,
)
from graph_longrange.gto_hardness import (
    HardnessMatrix
)

from macetools.electrostatics.utils import compute_total_charge_dipole, compute_qmmm_forces

@compile_mode("script")
class QEq(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        scale_atsize: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        self.scale_atsize = scale_atsize
        self.hidden_irreps = kwargs["hidden_irreps"]
        self.MLP_irreps = kwargs["MLP_irreps"]
        self.gate = kwargs["gate"]
        self.num_elements = kwargs["num_elements"]
        max_l = 0
        sigma = 1.0
        # electronetavitity block
        self.readouts_eneg = torch.nn.ModuleList()
        self.readouts_eneg.append(LinearReadoutBlock(self.hidden_irreps))
        for i in range(self.num_interactions - 1):
            if i == self.num_interactions - 2:
                hidden_irreps_out = str(
                    self.hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = self.hidden_irreps
            if i == self.num_interactions - 2:
                self.readouts_eneg.append(
                    NonLinearReadoutBlock(hidden_irreps_out, self.MLP_irreps, self.gate)
                )
            else:
                self.readouts_eneg.append(LinearReadoutBlock(self.hidden_irreps))

        self.eneg = LinearReadoutBlock(self.hidden_irreps)
        self.charge_equil = ChargeEquilibrationBlock(scale_atsize=self.scale_atsize)

        self.coulomb_energy = PBCAgnosticDirectElectrostaticEnergyBlock(
            density_max_l=max_l,  # max l multipoles, for now this is = 0 for real space stuff.
            density_smearing_width=sigma,
            kspace_cutoff=1.5* gto_basis_kspace_cutoff([sigma], max_l),  # 2.0 is <0.1meV/atom)
            include_pbc_corrections = True # MINE CHANGES
        )
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        use_pbc_evaluator: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        # if not training:
            # for p in self.parameters():
                # p.requires_grad = False
        # else:
            # for p in self.parameters():
                # p.requires_grad = True

        # print all trainable parameters of the network
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positionson 1"],
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
        edge_attrs = self.spherical_harmonics(vectors)
        # edge_feats = self.radial_embedding(lengths)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        # Interactions
        node_es_list = []
        node_eneg_list = []
        for interaction, product, readout, eneg in zip(
            self.interactions, self.products, self.readouts, self.readouts_eneg
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }
            node_eneg_list.append(eneg(node_feats).squeeze(-1))

        # with the final set of node features, predict the electronegativities using a charge equilibration scheme
        node_eneg = torch.sum(torch.stack(node_eneg_list, dim=0), dim=0).unsqueeze(-1)
        atomic_num_indices = torch.argmax(data["node_attrs"], dim=1)
        atomic_numbers = self.atomic_numbers[atomic_num_indices]
        node_partial_charges, elec_energy = self.charge_equil(data, node_eneg, atomic_numbers)
        

        # electro_energy = self.coulomb_energy(
        #     # charge_density,
        #     node_partial_charges.unsqueeze(-1),
        #     data["positions"],
        #     data["batch"],
        #     data["cell"].view(-1, 3, 3),
        #     data["rcell"].view(-1, 3, 3),
        #     data["volume"],
        #     data["pbc"].view(-1,3),
        #     num_graphs,
        #     use_pbc_evaluator=use_pbc_evaluator,
        # )

        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]'batch_dipoles

        total_energy = e0 + elec_energy 

        node_energy = node_e0 + node_inter_es
        forces, virials, stress = get_outputs(
            energy= total_energy,
            # energy=coulomb_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )
        # why is runnin the backward pass here changing the hardness tensor?
        total_charge, total_dipole = compute_total_charge_dipole(
                node_partial_charges.unsqueeze(-1),
                data["positions"],
                data["batch"],
                num_graphs
            )
        output = {
            "energy": total_energy,
            "forces": forces,
            "charges": node_partial_charges,
            "dipole": total_dipole,
            # "node_energy": node_energy,
            # "interaction_energy": inter_e,
            # "virials": virials,
            # "stress": stress,
            # "displacement": displacement,
        }
        return output

@compile_mode("script")
class maceQEq(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        scale_atsize: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        # print("scale_atsize",scale_atsize)
        self.scale_atsize = scale_atsize
        self.hidden_irreps = kwargs["hidden_irreps"]
        self.MLP_irreps = kwargs["MLP_irreps"]
        self.gate = kwargs["gate"]
        self.num_elements = kwargs["num_elements"]
        max_l = 0
        sigma = 1.0
        # electronegativity block
        self.readouts_eneg = torch.nn.ModuleList()
        self.readouts_eneg.append(LinearReadoutBlock(self.hidden_irreps))
        for i in range(self.num_interactions - 1):
            if i == self.num_interactions - 2:
                hidden_irreps_out = str(
                    self.hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = self.hidden_irreps
            if i == self.num_interactions - 2:
                self.readouts_eneg.append(
                    NonLinearReadoutBlock(hidden_irreps_out, self.MLP_irreps, self.gate)
                )
            else:
                self.readouts_eneg.append(LinearReadoutBlock(self.hidden_irreps))

        self.eneg = LinearReadoutBlock(self.hidden_irreps)
        self.charge_equil = ChargeEquilibrationBlock(scale_atsize=self.scale_atsize)

        self.coulomb_energy = PBCAgnosticDirectElectrostaticEnergyBlock(
            density_max_l=max_l,  # max l multipoles, for now this is = 0 for real space stuff.
            density_smearing_width=sigma,
            kspace_cutoff=1.5* gto_basis_kspace_cutoff([sigma], max_l),  # 2.0 is <0.1meV/atom)
            include_pbc_corrections = True # MINE CHANGES
        )
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        use_pbc_evaluator: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:

        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        # if not training:
            # for p in self.parameters():
                # p.requires_grad = False
        # else:
            # for p in self.parameters():
                # p.requires_grad = True

        # print all trainable parameters of the network
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positionson 1"],
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
        edge_attrs = self.spherical_harmonics(vectors)
        # edge_feats = self.radial_embedding(lengths)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        # Interactions
        node_es_list = []
        node_eneg_list = []
        for interaction, product, readout, eneg in zip(
            self.interactions, self.products, self.readouts, self.readouts_eneg
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }
            node_eneg_list.append(eneg(node_feats).squeeze(-1))

        # with the final set of node features, predict the electronegativities using a charge equilibration scheme
        node_eneg = torch.sum(torch.stack(node_eneg_list, dim=0), dim=0).unsqueeze(-1)
        atomic_num_indices = torch.argmax(data["node_attrs"], dim=1)
        atomic_numbers = self.atomic_numbers[atomic_num_indices]
        node_partial_charges, elec_energy = self.charge_equil(data,node_eneg,atomic_numbers)#, atomic_numbers)
        

        # electro_energy = self.coulomb_energy(
        #     # charge_density,
        #     node_partial_charges.unsqueeze(-1),
        #     data["positions"],
        #     data["batch"],
        #     data["cell"].view(-1, 3, 3),
        #     data["rcell"].view(-1, 3, 3),
        #     data["volume"],
        #     data["pbc"].view(-1,3),
        #     num_graphs,
        #     use_pbc_evaluator=use_pbc_evaluator,
        # )

        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]'batch_dipoles

        total_energy = e0 + elec_energy + inter_e

        node_energy = node_e0 + node_inter_es
        forces, virials, stress = get_outputs(
            energy= total_energy,
            # energy=coulomb_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )
        # why is runnin the backward pass here changing the hardness tensor?
        total_charge, total_dipole = compute_total_charge_dipole(
                node_partial_charges.unsqueeze(-1),
                data["positions"],
                data["batch"],
                num_graphs
            )
        output = {
            "energy": total_energy,
            "forces": forces,
            "charges": node_partial_charges,
            "dipole": total_dipole,
            # "node_energy": node_energy,
            # "interaction_energy": inter_e,
            # "virials": virials,
            # "stress": stress,
            # "displacement": displacement,
        }
        return output

@compile_mode("script")
class maceQEq_ESP(maceQEq):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        node_esp_irreps = o3.Irreps([(1, (0, 1))])
        node_feats_irreps = o3.Irreps([(self.hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.esp_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_esp_irreps, irreps_out=node_feats_irreps
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        use_pbc_evaluator: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:

        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        data["esp"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        # if not training:
            # for p in self.parameters():
                # p.requires_grad = False
        # else:
            # for p in self.parameters():
                # p.requires_grad = True

        # print all trainable parameters of the network
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positionson 1"],
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

        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        esp_data = data["esp"].unsqueeze(-1)

        # Embeddings
        esp_feats = self.esp_embedding(esp_data)
        node_feats = self.node_embedding(data["node_attrs"]) + esp_feats
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        # edge_feats = self.radial_embedding(lengths)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        # Interactions
        node_es_list = []
        node_eneg_list = []
        for interaction, product, readout, eneg in zip(
            self.interactions, self.products, self.readouts, self.readouts_eneg
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }
            node_eneg_list.append(eneg(node_feats).squeeze(-1))

        # with the final set of node features, predict the electronegativities using a charge equilibration scheme
        node_eneg = torch.sum(torch.stack(node_eneg_list, dim=0), dim=0).unsqueeze(-1) # shape [n_nodes, 1]
        node_eneg_esp = node_eneg + esp_data
        atomic_num_indices = torch.argmax(data["node_attrs"], dim=1)
        atomic_numbers = self.atomic_numbers[atomic_num_indices]
        node_partial_charges, elec_energy = self.charge_equil(data, node_eneg_esp, atomic_numbers)

        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]'batch_dipoles


        # Multiply the ESP data with the partial charges and sum over all nodes to get the QMMM energy
        e_qmmm = scatter_sum(
            src=node_partial_charges * esp_data[:, 0], # both shape [n_nodes, ] instead of [n_nodes, 1]
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs
        )
        total_energy = e0 + elec_energy + inter_e + e_qmmm

        qmmm_forces = compute_qmmm_forces(
                energy=total_energy,
                positions=data["positions"],
                esp=esp_data,
                esp_grad=data["esp_gradient"],
                compute_force=compute_force
        )

        node_energy = node_e0 + node_inter_es
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
        # why is runnin the backward pass here changing the hardness tensor?
        total_charge, total_dipole = compute_total_charge_dipole(
                node_partial_charges.unsqueeze(-1),
                data["positions"],
                data["batch"],
                num_graphs
            )
        
        if forces is not None:
            forces = forces + qmmm_forces

        output = {
            "energy": total_energy,
            "forces": forces,
            "charges": node_partial_charges,
            "dipole": total_dipole,
            # "node_energy": node_energy,
            # "interaction_energy": inter_e,
            # "virials": virials,
            # "stress": stress,
            # "displacement": displacement,
        }

        return output