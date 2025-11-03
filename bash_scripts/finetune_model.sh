#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH --mem=100GB
#SBATCH -p gpu-preempt
#SBATCH -G 1 # Number of GPUs
#SBATCH --constraint vram40
#SBATCH --nodes=1
#SBATCH --job-name=finetune_model
#SBATCH --output=./a_logs/finetune_model.out
#SBATCH --error=./a_logs/finetune_model.err
python -m scripts.finetune_model
                            