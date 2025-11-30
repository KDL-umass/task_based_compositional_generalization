#!/bin/bash

# Parameters
PROMPT_LENGTHS=("fixed")
PROMPT_MODES=("direct")
POS_EMBEDDING_TYPES=("abs" "rel_global")

TRAIN_SPLIT_STRATEGIES=("disjoint1_4_0" "disjoint1_4_20" "disjoint1_4_40" "disjoint1_4_60" "disjoint1_4_80" "disjoint1_4_100")
TRAIN_SPLIT_STRATEGIES+=("disjoint3_4_0" "disjoint3_4_20" "disjoint3_4_40" "disjoint3_4_60" "disjoint3_4_80" "disjoint3_4_100")
TRAIN_SPLIT_STRATEGIES+=("disjoint4_4_0" "disjoint4_4_20" "disjoint4_4_40" "disjoint4_4_60" "disjoint4_4_80" "disjoint4_4_100")
TRAIN_SPLIT_STRATEGIES+=("disjoint5_4_0" "disjoint5_4_20" "disjoint5_4_40" "disjoint5_4_60" "disjoint5_4_80" "disjoint5_4_100")
TRAIN_SPLIT_STRATEGIES+=("disjoint6_4_0" "disjoint6_4_20" "disjoint6_4_40" "disjoint6_4_60" "disjoint6_4_80" "disjoint6_4_100")
TRAIN_SPLIT_STRATEGIES+=("disjoint7_4_0" "disjoint7_4_20" "disjoint7_4_40" "disjoint7_4_60" "disjoint7_4_80" "disjoint7_4_100")
TRAIN_SPLIT_STRATEGIES+=("disjoint8_4_0" "disjoint8_4_20" "disjoint8_4_40" "disjoint8_4_60" "disjoint8_4_80" "disjoint8_4_100")
TRAIN_SPLIT_STRATEGIES+=("disjoint9_4_0" "disjoint9_4_20" "disjoint9_4_40" "disjoint9_4_60" "disjoint9_4_80" "disjoint9_4_100")

FUNCTION_TYPES=("diverse")
N_ALPHABETS=26
SEQ_LEN=6
N_FUNCTIONS=6
NHEADS_NLAYERS="nh6_nl3"
SEEDS=(0) 
EPOCHS=100
split_strategy_prefix="disjoint_4_diverse_fixed_all"
mkdir -p generated_jobs/${split_strategy_prefix}/
mkdir -p a_logs/training/${split_strategy_prefix}/

job_id=0
for split in "${TRAIN_SPLIT_STRATEGIES[@]}"; do
  for length in "${PROMPT_LENGTHS[@]}"; do
    for mode in "${PROMPT_MODES[@]}"; do
      for function_type in "${FUNCTION_TYPES[@]}"; do
        for pos_embedding_type in "${POS_EMBEDDING_TYPES[@]}"; do
          for seed in "${SEEDS[@]}"; do
            job_id=$((job_id + 1))
            TASK_MAX_LENGTH=$(echo "$split" | cut -d'_' -f2)
            
            job_file="generated_jobs/${split_strategy_prefix}/job_${job_id}_${split}_${pos_embedding_type}_${seed}.sh"
            
            cat <<EOF > "$job_file"
#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=${split}_${pos_embedding_type}_${function_type}_${seed}
#SBATCH --output=./a_logs/training/${split_strategy_prefix}/${split}_${pos_embedding_type}_${function_type}_${mode}_${length}_${seed}.out
#SBATCH --error=./a_logs/training/${split_strategy_prefix}/${split}_${pos_embedding_type}_${function_type}_${mode}_${length}_${seed}.err

module load conda/latest
cd /scratch4/workspace/ppruthi_umass_edu-CG/task_based_compositional_generalization
conda activate CG

python -m scripts.train_model \\
  --prompt_mode "$mode" \\
  --train_split "$split" \\
  --epochs "$EPOCHS" \\
  --pos_embedding_type "$pos_embedding_type" \\
  --n_heads_nlayers "$NHEADS_NLAYERS" \\
  --function_type "$function_type" \\
  --task_max_length "$TASK_MAX_LENGTH" \\
  --seed "$seed"
EOF

            chmod +x "$job_file"
          done
        done
      done
    done
  done
done

echo "Generated $job_id job files in ./generated_jobs/${split_strategy_prefix}/"
