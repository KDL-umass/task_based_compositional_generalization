#!/bin/bash
#SBATCH  -t 1:00:00
#SBATCH -p cpu
#SBATCH --mem 8GB
#SBATCH --cpus-per-task 1
#SBATCH --job-name=merge_livecodebench
#SBATCH --output=./a_logs/merge_livecodebench.out
#SBATCH --error=./a_logs/merge_livecodebench.err

module load conda/latest
cd /work/pi_pgrabowicz_umass_edu/awyuan/task_based_compositional_generalization
conda activate /work/pi_pgrabowicz_umass_edu/awyuan/conda_environments/cg-env

python -m scripts.merge_livecodebench_shards --num_shards 10 --holdout_size 96 --seed 0
