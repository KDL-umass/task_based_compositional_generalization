#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH --mem=100GB
#SBATCH -p gpu-preempt
#SBATCH -G 1 # Number of GPUs
#SBATCH --constraint vram40
#SBATCH --nodes=1
#SBATCH --job-name=evaluate_model_gemma1
#SBATCH --output=./a_logs/evaluate_model_gemma1.out
#SBATCH --error=./a_logs/evaluate_model_gemma1.err

module load conda/latest
cd /scratch4/workspace/ppruthi_umass_edu-CG/task_based_compositional_generalization
conda activate CG

N_ALPHABETS=26
SEQ_LEN=6
N_FUNCTIONS=6
NHEADS_NLAYERS="nh6_nl3"

# set task max length to 7 for identity-based train/test split
# set task max length to k in split strategy for without identity based train/test split
python -m scripts.evaluate_model \
    --prompt_mode "step_by_step" \
    --train_split "coverage_6_5_0" \
    --eval_split "coverage_6_5_0" \
    --nheads_nlayers "$NHEADS_NLAYERS" \
    --pos_embedding_type "rel_global" \
    --function_type "uniform" \
    --task_max_length 6 \
    --seed 0 
    

# # Command for representation analysis and equivalence class analysis to generate TSNE plots
# # keep strings representation analysis to True 
# # replace_strings_repr: True if you want to replace strings in representation analysis to a fixed string for ease of visualization of equivalence classes
# # replace_strings_repr : False if you want to keep the original train/test strings in representation analysis
# # keep equivalence_class_analysis to True if you want to compute equivalence class based on data and model
# # equivalence_class_type: both, data, model
# python -m scripts.evaluate_model \
#     --prompt_mode "direct" \
#     --prompt_length "fixed" \
#     --model_split "combination_3" \
#     --eval_split "combination_6" \
#     --nheads_nlayers "$NHEADS_NLAYERS" \
#     --n_alphabets "$N_ALPHABETS" \
#     --seq_len "$SEQ_LEN" \
#     --n_functions "$N_FUNCTIONS" \
#     --pos_embedding_type "rel_global" \
#     --function_type "uniform" \
#     --max_task_length 7 \
#     --seed 0 \
#     --repr_analysis True \
#     --replace_strings_repr "True" \
#     --equivalence_class_analysis True \
#     --equivalence_class_type "both"

