from ase.io import read,write
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def get_ref(
        mols,
        energy_keyword=None,
        forces_keyword=None,
        max_l = 0):
    ref_energy = []
    ref_forces = []
    ref_charges = []
    ref_DMA = []
    for m in mols:
        if energy_keyword != None:
            ref_energy.append(m.info[energy_keyword])
        if forces_keyword != None:
            ref_forces.extend(m.arrays[forces_keyword].flatten())
    ref_energy = np.array(ref_energy)
    ref_forces = np.array(ref_forces)
    ref_charges = np.array(ref_charges)
    ref_DMA = np.array(ref_DMA)
    return {"energy":ref_energy,"forces":ref_forces,"charges":ref_charges,"DMA":ref_DMA}

def get_MACE(
        mols,
        energy_keyword=None,
        forces_keyword=None,
        max_l = 0):
    ref_energy = []
    ref_forces = []
    ref_charges = []
    ref_DMA = []
    for m in mols:
        if energy_keyword != None:
            ref_energy.append(m.info[energy_keyword])
        if forces_keyword != None:
            ref_forces.extend(m.arrays[forces_keyword].flatten())
    ref_energy = np.array(ref_energy)
    ref_forces = np.array(ref_forces)
    ref_charges = np.array(ref_charges)
    ref_DMA = np.array(ref_DMA)
    return {"energy":ref_energy,"forces":ref_forces,"charges":ref_charges,"DMA":ref_DMA}


def compare_total_charge(
    mols,
    MACE_q_key = "MACE_charges",
    DFT_q_key = "aims_charges"):
    ref_charges = []
    mace_charges = [] 
    for m in mols:
        current_DFT = m.arrays[DFT_q_key]
        current_MACE = m.arrays[MACE_q_key]
        print("#########BEG#########")
        print("MACE: ",np.sum(current_MACE))
        print("DFT: ",np.sum(current_DFT))
        print("#########END#########")
        mace_charges.extend(current_MACE)
        ref_charges.extend(current_DFT)


mols = read("lala.xyz@:",format="extxyz")
# ref_data = get_ref(mols,"energy","forces","aims_charges","atomic_multipoles")
# MACE_data = get_MACE(mols,"MACE_energy","MACE_forces",None,"MACE_density_coefficients")

ref_data = get_ref(mols,"dft_energy","dft_forces")#,"atomic_multipoles")
MACE_data = get_MACE(mols,"MACE_energy","MACE_forces")
plot_charges = False 
plot_energy = True
plot_forces = True
plot_dma = False 

if plot_energy:
    plt.scatter(ref_data["energy"], MACE_data["energy"], c='blue', alpha=0.5, label='Data Points')  # Scatter plot
    plt.plot(ref_data["energy"],ref_data["energy"], color="black", label='Identity Line')  # Identity line
#    plt.title('MACE: NaCl clusters test structures')  # Title
    plt.xlabel('DFT energy')  # X-axis Label
    plt.ylabel('MACE energy')  # Y-axis Label
    plt.tight_layout()  # Tight layout for nicer appearance
    plt.savefig("MACEenergy.png",dpi=300)
    #plt.show()
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
    plt.title('MACE: NaCl clusters test structures')  # Title
    plt.xlabel('Hirshfeld charges')  # X-axis Label
    plt.ylabel('Mace Charges')  # Y-axis Label
    plt.tight_layout()  # Tight layout for nicer appearance
    plt.savefig("MACEcharges.png",dpi=300)
    #plt.show()
    plt.close()

if plot_forces:
    plt.scatter(ref_data["forces"], MACE_data["forces"], c='blue', alpha=0.5, label='Data Points')  # Scatter plot
    plt.plot(ref_data["forces"],ref_data["forces"], color="black", label='Identity Line')  # Identity line
    plt.title('MACE: NaCl clusters test structures')  # Title
    plt.xlabel('dft forces')  # X-axis Label
    plt.ylabel('mace forces')  # Y-axis Label
    plt.tight_layout()  # Tight layout for nicer appearance
    plt.savefig("MACEforces.png",dpi=300)
    #plt.show()
    plt.close()

# Calculate and print metrics
energy_mae = mean_absolute_error(ref_data["energy"], MACE_data["energy"])
energy_rmse = root_mean_squared_error(ref_data["energy"], MACE_data["energy"])
energy_r2 = r2_score(ref_data["energy"], MACE_data["energy"])

forces_mae = mean_absolute_error(ref_data["forces"], MACE_data["forces"])
forces_rmse = root_mean_squared_error(ref_data["forces"], MACE_data["forces"])
forces_r2 = r2_score(ref_data["forces"], MACE_data["forces"])

print(f'Energy MAE: {energy_mae:.4f} eV')
print(f'Energy RMSE: {energy_rmse:.4f} eV')
print(f'Energy R2: {energy_r2:.4f}')

print(f'Forces MAE: {forces_mae:.4f} eV/Å')
print(f'Forces RMSE: {forces_rmse:.4f} eV/Å')
print(f'Forces R2: {forces_r2:.4f}')

