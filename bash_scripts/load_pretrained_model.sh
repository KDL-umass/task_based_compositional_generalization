#!/bin/bash
#SBATCH  -t 1:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --constraint vram16
#SBATCH --cpus-per-task 4
#SBATCH --job-name=load_pretrained_model
#SBATCH --output=./a_logs/load_pretrained_model.out
#SBATCH --error=./a_logs/load_pretrained_model.err
module load conda/latest
cd /scratch4/workspace/ppruthi_umass_edu-CG/task_based_compositional_generalization
conda activate CG
python -m src.models.pretrained 

