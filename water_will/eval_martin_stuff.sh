#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=eval.out
#SBATCH --error=eval.err
#SBATCH --gres=gpu:1

export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools
export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange

python /lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools/scripts/eval_qeq.py \
    --configs="water100.xyz" \
    --model="QEqWwater.model" \
    --output="pol_martin_res.xyz" \
    --device="cuda" 
