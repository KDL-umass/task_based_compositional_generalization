#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -p cpu
#SBATCH -q long
#SBATCH --mem 64GB
#SBATCH --cpus-per-task 1
#SBATCH --job-name=gen_livecodebench_nontrained_test
#SBATCH --output=./a_logs/generate_livecodebench_nontrained_test.out
#SBATCH --error=./a_logs/generate_livecodebench_nontrained_test.err

export HF_HOME=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/cache/huggingface
export HF_DATASETS_CACHE=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/datasets

module load conda/latest
cd /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization
conda activate /work/pi_pgrabowicz_umass_edu/awyuan/conda_environments/cg-env

FUNCTIONS_JSON=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/data/livecodebench/filtered_functions.json
OUTPUT_DIR=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/data/livecodebench/heldout_4_200/seed_0

python -m scripts.generate_livecodebench_filtered_functions_test \
  --function_names_json ${FUNCTIONS_JSON} \
  --per_function 200 \
  --seed 0 \
  --output_dir ${OUTPUT_DIR}
