#!/bin/bash
#SBATCH  -t 12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --constraint "vram32&sm_70"
#SBATCH --cpus-per-task 1
#SBATCH --job-name=eval_livecodebench_gemma1
#SBATCH --output=./a_logs/eval_livecodebench_gemma1.out
#SBATCH --error=./a_logs/eval_livecodebench_gemma1.err

export HF_HOME=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/cache/huggingface
# export HF_HOME=/datasets/ai/llama3
mkdir -p /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/cache/huggingface
cp ~/.cache/huggingface/token /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/cache/huggingface/token
echo $HF_HOME
module load conda/latest
cd /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization
conda activate /work/pi_pgrabowicz_umass_edu/awyuan/conda_environments/cg-env

python -m scripts.evaluate_livecodebench --model_name gemma1 --split test
