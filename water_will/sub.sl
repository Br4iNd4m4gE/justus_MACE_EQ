#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J mace_old 
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
#
#SBATCH --time=12:00:00

module load anaconda/3/2023.03
module load cuda/12.1 cudnn/8.9.2
module load intel/21.2.0 impi/2021.2 mkl/2022.2 
source ${MKLROOT}/env/vars.sh intel64
module load pytorch
# module purge
# module load intel/21.2.0 impi/2021.2 cuda/11.2
# conda activate mace_env 
export PYTHONPATH=${PYTHONPATH}:/u/mvondrak/mace_qeq_dev/mace-tools
export PYTHONPATH=${PYTHONPATH}:/u/mvondrak/mace_qeq_dev/graph_longrange

python qeq_train.py --name="NaClWaterWill" --train_file="water_will.xyz" --batch_size=40 --valid_fraction=0.05 --config_type_weights='{"Default":1.0}' --E0s='{1: -12.6746244439181, 8: -2041.03979050724}' --model="FullQEqMace" --hidden_irreps='64x0e' --r_max=8.0 --max_num_epochs=2000 --device=cuda --loss="charges_energy_forces" --formal_charges_from_data --charges_key="aims_charges" --error_table="MartinCharges" --scale_atsize=1.0 --restart_latest >> output 
