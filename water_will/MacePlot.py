#!/lustre/home/ka/ka_ipc/ka_he8978/miniconda3/envs/mace_env/bin/python3.12
from ase.io import read,write
import numpy as np
import matplotlib.pyplot as plt

REF_GEOMS = "water100.xyz"


def get_ref(
        mols,
        energy_keyword=None,
        forces_keyword=None,
        charges_keyword=None,
        DMA_keyword=None,
        max_l = 0):
    ref_energy = []
    ref_forces = []
    ref_charges = []
    ref_DMA = []
    for m in mols:
        if charges_keyword != None:
            ref_charges.extend(m.arrays[charges_keyword])
        if energy_keyword != None:
            if energy_keyword == "energy":
                ref_energy.append(m.get_potential_energy())
            else:
                ref_energy.append(m.info[energy_keyword])
        if forces_keyword != None:
            if forces_keyword == "forces":
                ref_forces.extend(m.get_forces().flatten())
            else:
                ref_forces.extend(m.arrays[forces_keyword].flatten())
        if DMA_keyword != None:
            AIMS_atom_multipoles = m.arrays[DMA_keyword]
            ref_DMA.extend(AIMS_atom_multipoles[:,0])
    ref_energy = np.array(ref_energy)
    ref_forces = np.array(ref_forces)
    ref_charges = np.array(ref_charges)
    ref_DMA = np.array(ref_DMA)
    return {"energy":ref_energy,"forces":ref_forces,"charges":ref_charges,"DMA":ref_DMA}

def get_MACE(
        mols,
        energy_keyword=None,
        forces_keyword=None,
        charges_keyword=None,
        DMA_keyword=None,
        max_l = 0):
    ref_energy = []
    ref_forces = []
    ref_charges = []
    ref_DMA = []
    for m in mols:
        if charges_keyword != None:
            ref_charges.extend(m.arrays[charges_keyword])
        if energy_keyword != None:
            if energy_keyword == "energy":
                ref_energy.append(m.get_potential_energy())
            else:
                ref_energy.append(m.info[energy_keyword])
        if forces_keyword != None:
            if forces_keyword == "forces":
                ref_forces.extend(m.get_forces().flatten())
            else:
                ref_forces.extend(m.arrays[forces_keyword].flatten())
        if DMA_keyword != None:
            AIMS_atom_multipoles = m.arrays[DMA_keyword]
            ref_DMA.extend(AIMS_atom_multipoles)
    ref_energy = np.array(ref_energy)
    ref_forces = np.array(ref_forces)
    ref_charges = np.array(ref_charges)
    ref_DMA = np.array(ref_DMA)
    return {"energy":ref_energy,"forces":ref_forces,"charges":ref_charges,"DMA":ref_DMA}

mols = read("water100.xyz",format="extxyz", index=":")
ref_data = get_ref(mols,"energy","forces","aims_charges",None)
mols = read("pol_martin_res.xyz",format="extxyz", index=":")
MACE_data = get_MACE(mols,"MACE_energy","MACE_forces","MACE_charges",None)

plot_charges = True 
plot_energy = True
plot_forces = True
plot_dma = False 

if plot_energy:
    plt.scatter(ref_data["energy"], MACE_data["energy"], c='blue', alpha=0.5, label='Data Points')  # Scatter plot
    plt.plot(ref_data["energy"],ref_data["energy"], color="black", label='Identity Line')  # Identity line
    plt.xlabel('DFT energy')  # X-axis Label
    plt.ylabel('MACE energy')  # Y-axis Label
    plt.tight_layout()  # Tight layout for nicer appearance
    plt.savefig("MACEenergy.png",dpi=300)
    # plt.show()
    plt.close()

if plot_dma:
    plt.scatter(ref_data["DMA"], MACE_data["DMA"], c='blue', alpha=0.5, label='Data Points')  # Scatter plot
    plt.plot(ref_data["DMA"],ref_data["DMA"], color="black", label='Identity Line')  # Identity line
    plt.xlabel('DMA ref')  # X-axis Label
    plt.ylabel('Mace DMA')  # Y-axis Label
    plt.tight_layout()  # Tight layout for nicer appearance
    plt.savefig("MACEdma.png",dpi=300)
    #plt.show()
    plt.close()

    
print(len(ref_data["charges"]),len(MACE_data["charges"]))    
if plot_charges:
    plt.scatter(ref_data["charges"], MACE_data["charges"], c='blue', alpha=0.5, label='Data Points')  # Scatter plot
    plt.plot(ref_data["charges"],ref_data["charges"], color="black", label='Identity Line')  # Identity line
    plt.xlabel('Hirshfeld charges')  # X-axis Label
    plt.ylabel('Mace Charges')  # Y-axis Label
    plt.tight_layout()  # Tight layout for nicer appearance
    plt.savefig("MACEcharges.png",dpi=300)
    #plt.show()
    plt.close()

if plot_forces:
    plt.scatter(ref_data["forces"], MACE_data["forces"], c='blue', alpha=0.5, label='Data Points')  # Scatter plot
    plt.plot(ref_data["forces"],ref_data["forces"], color="black", label='Identity Line')  # Identity line
    plt.xlabel('dft forces')  # X-axis Label
    plt.ylabel('mace forces')  # Y-axis Label
    plt.tight_layout()  # Tight layout for nicer appearance
    plt.savefig("MACEforces.png",dpi=300)
    #plt.show()
    plt.close()

