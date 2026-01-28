#!/bin/bash
#SBATCH  -t 2-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --constraint "vram40&sm_70"
# #SBATCH --constraint a100
#SBATCH --cpus-per-task 1
#SBATCH --job-name=finetune_livecodebench_lora_1_0
#SBATCH --output=./a_logs/finetune_livecodebench_lora_1_0.out
#SBATCH --error=./a_logs/finetune_livecodebench_lora_1_0.err

export HF_HOME=/datasets/ai/llama3
export HF_DATASETS_CACHE=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/datasets

module load conda/latest
cd /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization
conda activate /work/pi_pgrabowicz_umass_edu/awyuan/conda_environments/cg-env

python -m scripts.finetune_livecodebench_lora
