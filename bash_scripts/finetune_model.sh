#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -c 32
#SBATCH --mem=500G
#SBATCH -p gpu-preempt
#SBATCH -G 2 # Number of GPUs
#SBATCH --constraint a100
#SBATCH --nodes=1
#SBATCH --job-name=finetune_model
#SBATCH --output=./a_logs/finetune_model.out
#SBATCH --error=./a_logs/finetune_model.err

# Fine-tuning configurations
PROMPT_LENGTHS=("fixed")
PROMPT_MODES=("direct")
FUNCTION_TYPES=("uniform")
MODELS=("llama3")  # Can use "gpt2", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-3.1-8b"

# Training hyperparameters
EPOCHS=3
N_ALPHABETS=26
SEQ_LEN=6
N_FUNCTIONS=6
SEEDS=(0)

# Train split strategies
TRAIN_SPLIT_STRATEGIES=("combination_6")

# Fine-tuning specific hyperparameters
BATCH_SIZE=1
LEARNING_RATE=5e-5
GRADIENT_ACCUMULATION_STEPS=1
WARMUP_STEPS=100
SAVE_STEPS=500
EVAL_STEPS=500


echo "=== FINE-TUNING PRETRAINED MODELS ==="

for model in "${MODELS[@]}"; do
    for split in "${TRAIN_SPLIT_STRATEGIES[@]}"; do
        for length in "${PROMPT_LENGTHS[@]}"; do
            for mode in "${PROMPT_MODES[@]}"; do
                for function_type in "${FUNCTION_TYPES[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        # Extract task max length from split strategy
                        TASK_MAX_LENGTH=$(echo "$split" | cut -d'_' -f2)
                        
                        echo "========================================="
                        echo "Model: $model"
                        echo "Mode: $mode - Length: $length - Split: $split"
                        echo "Task max length: $TASK_MAX_LENGTH"
                        echo "Epochs: $EPOCHS, Batch size: $BATCH_SIZE"
                        echo "Learning rate: $LEARNING_RATE"
                        echo "========================================="
                        
                        python -m scripts.finetune_model \
                            --model "$model" \
                            --prompt_mode "$mode" \
                            --train_split "$split" \
                            --epochs "$EPOCHS" \
                            --n_alphabets "$N_ALPHABETS" \
                            --seq_len "$SEQ_LEN" \
                            --n_functions "$N_FUNCTIONS" \
                            --function_type "$function_type" \
                            --task_max_length "$TASK_MAX_LENGTH" \
                            --batch_size "$BATCH_SIZE" \
                            --learning_rate "$LEARNING_RATE" \
                            --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
                            --warmup_steps "$WARMUP_STEPS" \
                            --save_steps "$SAVE_STEPS" \
                            --eval_steps "$EVAL_STEPS" \
                            --seed "$seed" \
                            --torch_dtype "bfloat16" \
                            --bf16

                        echo "Completed: $model - $mode - $length - $split"
                        echo ""
                    done
                done
            done
        done
    done
done

echo "=== FINE-TUNING COMPLETED ==="

