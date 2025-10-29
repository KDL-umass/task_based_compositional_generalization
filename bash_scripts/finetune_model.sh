#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -c 32
#SBATCH --mem=500G
#SBATCH -p gpu-preempt
#SBATCH -G 2 # Number of GPUs
#SBATCH --constraint a100
#SBATCH --nodes=1
#SBATCH --job-name=finetune_model
#SBATCH --output=./a_logs/finetune_model.out
#SBATCH --error=./a_logs/finetune_model.err

python -m scripts.finetune_model
                            