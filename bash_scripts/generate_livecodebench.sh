#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH -p cpu
#SBATCH -q long
#SBATCH --mem 64GB
#SBATCH --cpus-per-task 1
#SBATCH --job-name=gen_livecodebench
#SBATCH --array=0-9
#SBATCH --output=./a_logs/generate_livecodebench_%a.out
#SBATCH --error=./a_logs/generate_livecodebench_%a.err

export HF_HOME=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/cache/huggingface
export HF_DATASETS_CACHE=/work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization/datasets

module load conda/latest
cd /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization
conda activate /work/pi_pgrabowicz_umass_edu/awyuan/conda_environments/cg-env

python -m scripts.generate_livecodebench --num_shards 10 --shard_id ${SLURM_ARRAY_TASK_ID}
