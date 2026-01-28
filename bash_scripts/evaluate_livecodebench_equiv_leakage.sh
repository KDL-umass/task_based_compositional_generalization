#!/bin/bash
#SBATCH  -t 12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --constraint "vram40&sm_70"
#SBATCH --cpus-per-task 1
#SBATCH --job-name=eval_livecodebench_llama3_equiv_0.0
#SBATCH --output=./a_logs/eval_livecodebench_llama3_equiv_0.0.out
#SBATCH --error=./a_logs/eval_livecodebench_llama3_equiv_0.0.err

export HF_HOME=/datasets/ai/llama3
mkdir -p /datasets/ai/llama3
cp ~/.cache/huggingface/token /datasets/ai/llama3/token

echo $HF_HOME
module load conda/latest
cd /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization
conda activate /work/pi_pgrabowicz_umass_edu/awyuan/conda_environments/cg-env

python -m scripts.evaluate_livecodebench_equiv_leakage \
  --model_name llama3 \
  --data_dir /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/data/livecodebench/equiv_leakage_shared_0.0/holdout_54/seed_0 \
  --lora_path /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/models/ckpts/livecodebench_lora/equiv_leakage_shared_0.0/holdout_54/seed_0/model_llama3/lora_r16_a32/final