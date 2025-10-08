#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=eval.out
#SBATCH --error=eval.err
#SBATCH --gres=gpu:1

mace_eval_configs \
    --configs="train_set_smoll.xyz" \
    --model="water_model_swa.model" \
    --output="lala.xyz"
