#!/bin/bash
#SBATCH  -t 12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --constraint a40
#SBATCH --cpus-per-task 4
#SBATCH --job-name=load_pretrained_model
#SBATCH --output=./a_logs/load_pretrained_model.out
#SBATCH --error=./a_logs/load_pretrained_model.err
export HF_HOME=/datasets/ai/llama3
echo $HF_HOME
module load conda/latest
cd /project/pi_jensen_umass_edu/ppruthi_umass_edu/task_based_compositional_generalization
conda activate CG
python -m src.models.pretrained $HF_HOME

