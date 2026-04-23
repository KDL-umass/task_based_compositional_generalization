#!/bin/bash

# Parameters
PROMPT_LENGTHS=("fixed")
PROMPT_MODES=("direct" "step_by_step")
POS_EMBEDDING_TYPES=("rel_global")

TRAIN_SPLIT_STRATEGIES=("combination_2")

FUNCTION_TYPES=("diverse")
N_ALPHABETS=26
SEQ_LEN=6
N_FUNCTIONS=6
NHEADS_NLAYERS="nh6_nl3"
SEEDS=(0) 
EPOCHS=100
NSAMPLES=(2000 4000)
split_strategy_prefix="sample_efficiency_2"
mkdir -p generated_jobs/${split_strategy_prefix}/
mkdir -p a_logs/training/${split_strategy_prefix}/

job_id=0
for split in "${TRAIN_SPLIT_STRATEGIES[@]}"; do
  for length in "${PROMPT_LENGTHS[@]}"; do
    for mode in "${PROMPT_MODES[@]}"; do
      for function_type in "${FUNCTION_TYPES[@]}"; do
        for pos_embedding_type in "${POS_EMBEDDING_TYPES[@]}"; do
          for seed in "${SEEDS[@]}"; do
            for nsamples in "${NSAMPLES[@]}"; do
              job_id=$((job_id + 1))
              TASK_MAX_LENGTH=$(echo "$split" | cut -d'_' -f2)
              
              job_file="generated_jobs/${split_strategy_prefix}/job_${job_id}_${split}_${pos_embedding_type}_${function_type}_${mode}_${length}_${seed}_${nsamples}.sh"
              
              cat <<EOF > "$job_file"
#!/bin/bash
#SBATCH -t 5:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=${split}_${pos_embedding_type}_${function_type}_${mode}_${length}_${seed}_${nsamples}
#SBATCH --output=./a_logs/training/${split_strategy_prefix}/${split}_${pos_embedding_type}_${function_type}_${mode}_${length}_${seed}_${nsamples}.out
#SBATCH --error=./a_logs/training/${split_strategy_prefix}/${split}_${pos_embedding_type}_${function_type}_${mode}_${length}_${seed}_${nsamples}.err

module load conda/latest
cd /scratch/workspace/ppruthi_umass_edu-MI/task_based_compositional_generalization
conda activate CG

python -m scripts.train_model \\
  --prompt_mode "$mode" \\
  --train_split "$split" \\
  --epochs "$EPOCHS" \\
  --pos_embedding_type "$pos_embedding_type" \\
  --n_heads_nlayers "$NHEADS_NLAYERS" \\
  --function_type "$function_type" \\
  --task_max_length "$TASK_MAX_LENGTH" \\
  --seed "$seed" \\
  --sample_efficiency_experiment True \\
  --nsamples "$nsamples"
EOF

              chmod +x "$job_file"
            done
          done
        done
      done
    done
  done
done
echo "Generated $job_id job files in ./generated_jobs/${split_strategy_prefix}/"
