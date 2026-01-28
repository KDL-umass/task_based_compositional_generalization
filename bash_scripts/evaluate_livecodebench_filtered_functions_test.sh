#!/bin/bash
#SBATCH  -t 12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --constraint "vram40&sm_70"
#SBATCH --cpus-per-task 1
#SBATCH --job-name=eval_livecodebench_llama3_filtered_test
#SBATCH --array=0-1
#SBATCH --output=./a_logs/eval_livecodebench_llama3_filtered_test_%a.out
#SBATCH --error=./a_logs/eval_livecodebench_llama3_filtered_test_%a.err

export HF_HOME=/datasets/ai/llama3
mkdir -p /datasets/ai/llama3
cp ~/.cache/huggingface/token /datasets/ai/llama3/token

echo $HF_HOME
module load conda/latest
cd /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization
conda activate /work/pi_pgrabowicz_umass_edu/awyuan/conda_environments/cg-env

SHARED_FRACTIONS=(0.0 1.0)
SHARED_FRACTION=${SHARED_FRACTIONS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/data/livecodebench/heldout_4_200/seed_0
LORA_DIR=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/models/ckpts/livecodebench_lora/equiv_leakage_shared_${SHARED_FRACTION}/holdout_54/seed_0/model_llama3/lora_r16_a32/final

python -m scripts.evaluate_livecodebench_equiv_leakage \
  --model_name llama3 \
  --data_dir ${DATA_DIR} \
  --results_subdir shared_${SHARED_FRACTION} \
  --lora_path ${LORA_DIR}
