#!/bin/bash
#SBATCH  -t 4:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --cpus-per-task 4
#SBATCH --nodes=1
#SBATCH --job-name=visualize_representations
#SBATCH --output=./a_logs/visualize_representations.out
#SBATCH --error=./a_logs/visualize_representations.err

module load conda/latest
cd /scratch/workspace/ppruthi_umass_edu-MI/task_based_cg_debugging/task_based_compositional_generalization
conda activate CG

python -m scripts.visualize_representations