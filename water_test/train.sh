#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --gres=gpu:1
#SBATCH --mail-user=lukas.petersen@kit.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --open-mode=append

mace_run_train \
    --name="water_model" \
    --train_file="train_set_smoll.xyz" \
    --valid_file="train_set_smoll.xyz" \
    --test_file="test_set.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --E0s='{1: -12.6746244439181, 8: -2041.03979050724}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.0 \
    --batch_size=10 \
    --max_num_epochs=1000 \
    --swa \
    --start_swa=450 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --forces_key="dft_forces" \
    --energy_key="dft_energy" \
    --device=cuda \
