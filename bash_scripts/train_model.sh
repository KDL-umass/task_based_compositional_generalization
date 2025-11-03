#!/bin/bash
#SBATCH  -t 16:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --cpus-per-task 4
#SBATCH --job-name=train_model
#SBATCH --output=./a_logs/train_model.out
#SBATCH --error=./a_logs/train_model.err
module load conda/latest
cd /work/pi_jensen_umass_edu/ppruthi_umass_edu/task_based_compositional_generalization
conda activate CG

PROMPT_LENGTHS=("fixed")
PROMPT_MODES=("direct" "step_by_step")
POS_EMBEDDING_TYPES=("abs" "rel_global")
FUNCTION_TYPES=("uniform" "diverse")


EPOCHS=100
N_ALPHABETS=26
SEQ_LEN=6
N_FUNCTIONS=6
NHEADS_NLAYERS="nh12_nl12"
SEEDS=(0)

TRAIN_SPLIT_STRATEGIES=("combination_6")

# task max length is k_max and gets k from the split strategy without identity modules. Fix task_max_length to 7 for identity-based train/test split
echo "=== TRAINING ==="
for split in "${TRAIN_SPLIT_STRATEGIES[@]}"; do
    for length in "${PROMPT_LENGTHS[@]}"; do
        for mode in "${PROMPT_MODES[@]}"; do
            for function_type in "${FUNCTION_TYPES[@]}"; do
                for pos_embedding_type in "${POS_EMBEDDING_TYPES[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        echo "Training: $mode - $length - $split"
                        # get task max length from split
                        TASK_MAX_LENGTH=$(echo "$split" | cut -d'_' -f2)
                        echo "Task max length: $TASK_MAX_LENGTH"
                        python -m scripts.train_model \
                            --prompt_mode "$mode" \
                            --train_split "$split" \
                            --epochs "$EPOCHS" \
                            --pos_embedding_type "$pos_embedding_type" \
                            --n_heads_nlayers "$NHEADS_NLAYERS" \
                            --function_type "$function_type" \
                            --task_max_length "$TASK_MAX_LENGTH" 
                    done
                done
            done
        done
    done
done



