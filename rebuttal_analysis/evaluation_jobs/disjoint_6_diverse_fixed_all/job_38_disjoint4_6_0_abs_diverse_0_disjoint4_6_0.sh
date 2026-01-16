#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=disjoint4_6_0_abs_diverse_0_disjoint4_6_0
#SBATCH --output=./a_logs/evaluation/disjoint_6_diverse_fixed_all/disjoint4_6_0_abs_diverse_direct_0_disjoint4_6_0.out
#SBATCH --error=./a_logs/evaluation/disjoint_6_diverse_fixed_all/disjoint4_6_0_abs_diverse_direct_0_disjoint4_6_0.err

module load conda/latest
cd /scratch4/workspace/ppruthi_umass_edu-CG/task_based_compositional_generalization
conda activate CG

python -m scripts.evaluate_model \
  --prompt_mode "direct" \
  --train_split "disjoint4_6_0" \
  --eval_split "disjoint4_6_0" \
  --nheads_nlayers "nh6_nl3" \
  --pos_embedding_type "abs" \
  --function_type "diverse" \
  --task_max_length "6" \
  --seed "0" \
