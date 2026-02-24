#!/bin/bash
#SBATCH  -t 2-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --constraint "vram40"
#SBATCH --cpus-per-task 1
#SBATCH --job-name=finetune_model_gemma1_disjoint2_6_25_diverse    
#SBATCH --output=./a_logs/finetune_model_gemma1_disjoint2_6_25_diverse.out
#SBATCH --error=./a_logs/finetune_model_gemma1_disjoint2_6_25_diverse.err
module load conda/latest
cd /scratch4/workspace/ppruthi_umass_edu-CG/task_based_compositional_generalization
conda activate CG

python -m scripts.finetune_model
                            