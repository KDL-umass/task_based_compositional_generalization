#!/bin/bash
#SBATCH  -t 1:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --constraint vram40
#SBATCH --cpus-per-task 4
#SBATCH --job-name=load_pretrained_model
#SBATCH --output=./a_logs/load_pretrained_model.out
#SBATCH --error=./a_logs/load_pretrained_model.err
export HF_HOME=/datasets/ai/llama3
echo $HF_HOME
module load conda/latest
cd /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization
conda activate /work/pi_pgrabowicz_umass_edu/awyuan/conda_environments/cg-env
python -m src.models.pretrained $HF_HOME

