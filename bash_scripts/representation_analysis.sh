#!/bin/bash
#SBATCH  -t 16:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --cpus-per-task 4
#SBATCH --nodes=1
#SBATCH --job-name=representation_analysis_2_step_by_step
#SBATCH --output=./a_logs/rep_analysis_2_step_by_step.out
#SBATCH --error=./a_logs/rep_analysis_2_step_by_step.err

module load conda/latest
cd /scratch/workspace/ppruthi_umass_edu-MI/task_based_cg_debugging/task_based_compositional_generalization
conda activate CG

N_ALPHABETS=26
SEQ_LEN=6
N_FUNCTIONS=6
NHEADS_NLAYERS="nh6_nl3"

# set task max length to 7 for identity-based train/test split
# set task max length to k in split strategy for without identity based train/test split
python -m scripts.analyze_representations \
    --prompt_mode "step_by_step" \
    --train_split "combination_2" \
    --eval_split "combination_2" \
    --nheads_nlayers "$NHEADS_NLAYERS" \
    --pos_embedding_type "rel_global" \
    --function_type "uniform" \
    --task_max_length 2 \
    --seed 0 