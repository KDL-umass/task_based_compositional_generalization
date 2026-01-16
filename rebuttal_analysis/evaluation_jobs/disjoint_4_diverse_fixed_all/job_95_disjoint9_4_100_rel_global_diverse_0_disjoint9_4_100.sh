#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=disjoint9_4_100_rel_global_diverse_0_disjoint9_4_100
#SBATCH --output=./a_logs/evaluation/disjoint_4_diverse_fixed_all/disjoint9_4_100_rel_global_diverse_direct_0_disjoint9_4_100.out
#SBATCH --error=./a_logs/evaluation/disjoint_4_diverse_fixed_all/disjoint9_4_100_rel_global_diverse_direct_0_disjoint9_4_100.err

module load conda/latest
cd /scratch4/workspace/ppruthi_umass_edu-CG/task_based_compositional_generalization
conda activate CG

python -m scripts.evaluate_model \
  --prompt_mode "direct" \
  --train_split "disjoint9_4_100" \
  --eval_split "disjoint9_4_100" \
  --nheads_nlayers "nh6_nl3" \
  --pos_embedding_type "rel_global" \
  --function_type "diverse" \
  --task_max_length "4" \
  --seed "0" \
