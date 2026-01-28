#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH -p cpu
#SBATCH -q long
#SBATCH --mem 64GB
#SBATCH --cpus-per-task 1
#SBATCH --job-name=gen_livecodebench_leak
#SBATCH --array=0-11
#SBATCH --output=./a_logs/generate_livecodebench_leak_%a.out
#SBATCH --error=./a_logs/generate_livecodebench_leak_%a.err

export HF_HOME=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/cache/huggingface
export HF_DATASETS_CACHE=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/datasets

module load conda/latest
cd /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization
conda activate /work/pi_pgrabowicz_umass_edu/awyuan/conda_environments/cg-env

FILTERED_IDS_JSON=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/data/livecodebench/livecodebench_filtered_ids.json
HELDOUT_FUNCTIONS_JSON=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/data/livecodebench/heldout_functions.json

SHARED_FRACTIONS=(0 0.5 1)
SHARED_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
SHARD_ID=$((SLURM_ARRAY_TASK_ID % 4))
SHARED_FRACTION=${SHARED_FRACTIONS[$SHARED_INDEX]}

python -m scripts.generate_livecodebench \
  --num_shards 4 \
  --shard_id ${SHARD_ID} \
  --mode equiv_leakage \
  --holdout_size 54 \
  --filtered_ids_json ${FILTERED_IDS_JSON} \
  --heldout_function_names_json ${HELDOUT_FUNCTIONS_JSON} \
  --shared_fraction ${SHARED_FRACTION}
