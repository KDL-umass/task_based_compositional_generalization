#!/bin/bash
#SBATCH  -t 1:00:00
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --cpus-per-task 4
#SBATCH --job-name=analysis_reversepaircoverage_6_0
#SBATCH --output=./a_logs/analysis_reversepaircoverage_6_0.out
#SBATCH --error=./a_logs/analysis_reversepaircoverage_6_0.err
module load conda/latest
cd /scratch4/workspace/ppruthi_umass_edu-CG/task_based_compositional_generalization
conda activate CG

python -m src.analysis.analyzer \
    --prompt_mode "step_by_step" \
    --train_split "reversepaircoverage_6_0" \
    --eval_split "reversepaircoverage_6_0" \
    --nheads_nlayers "nh6_nl3" \
    --pos_embedding_type "rel_global" \
    --function_type "uniform" \
    --seed 0 \
    --task_max_length 6

python -m scripts.representation_visualization \
    --prompt_mode "step_by_step" \
    --train_split "reversepaircoverage_6_0" \
    --eval_split "reversepaircoverage_6_0" \
    --nheads_nlayers "nh6_nl3" \
    --pos_embedding_type "rel_global" \
    --function_type "uniform" \
    --seed 0 \
    --task_max_length 6