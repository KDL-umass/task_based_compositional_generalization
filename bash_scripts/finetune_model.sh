#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH --mem=100GB
#SBATCH -p gpu
#SBATCH -G 1 # Number of GPUs
#SBATCH --constraint a40
#SBATCH --nodes=1
#SBATCH --job-name=finetune_model_gemma1_disjoint7_6_60_diverse    
#SBATCH --output=./a_logs/finetune_model_gemma1_disjoint7_6_60_diverse.out
#SBATCH --error=./a_logs/finetune_model_gemma1_disjoint7_6_60_diverse.err
module load conda/latest
cd /scratch4/workspace/ppruthi_umass_edu-CG/task_based_compositional_generalization
conda activate CG

python -m scripts.finetune_model
                            