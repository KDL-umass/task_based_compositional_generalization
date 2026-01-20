#!/bin/bash
#SBATCH  -t 12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --constraint "vram8&sm_70"
#SBATCH --cpus-per-task 1
#SBATCH --job-name=eval_livecodebench_granite
#SBATCH --output=./a_logs/eval_livecodebench_granite.out
#SBATCH --error=./a_logs/eval_livecodebench_granite.err

export HF_HOME=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/cache/huggingface
echo $HF_HOME
module load conda/latest
cd /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization
conda activate /work/pi_pgrabowicz_umass_edu/awyuan/conda_environments/cg-env

python -m scripts.evaluate_livecodebench --model_name granite --split test
