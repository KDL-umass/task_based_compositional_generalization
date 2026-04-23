#!/bin/bash

# Parameters
PROMPT_LENGTHS=("fixed")
PROMPT_MODES=("direct")
POS_EMBEDDING_TYPES=("rel_global" "abs")
PROMPT_MODES=("direct" "step_by_step")
POS_EMBEDDING_TYPES=("rel_global")


TRAIN_SPLIT_STRATEGIES=("disjoint7_6_100" "disjoint7_6_70" "disjoint7_6_60" "disjoint7_6_50" "disjoint7_6_20" "disjoint7_6_10" "disjoint7_6_0")

FUNCTION_TYPES=("uniform")
N_ALPHABETS=26
SEQ_LEN=6
N_FUNCTIONS=6
NHEADS_NLAYERS="nh6_nl3"
SEEDS=(0)
split_strategy_prefix="disjoint7_6_diverse_nh6_nl3"
split_strategy_prefix="sample_efficiency_2"
NSAMPLES=(200 500 1000)

mkdir -p evaluation_jobs/${split_strategy_prefix}/
mkdir -p a_logs/evaluation/${split_strategy_prefix}/

job_id=0
for split in "${TRAIN_SPLIT_STRATEGIES[@]}"; do
  for length in "${PROMPT_LENGTHS[@]}"; do
    for mode in "${PROMPT_MODES[@]}"; do
      for function_type in "${FUNCTION_TYPES[@]}"; do
        for pos_embedding_type in "${POS_EMBEDDING_TYPES[@]}"; do
          for nsamples in "${NSAMPLES[@]}"; do
            for eval_split in "${TRAIN_SPLIT_STRATEGIES[@]}"; do
          # skip eval_split if it is not the same as split
            if [ "$eval_split" != "$split" ]; then
              continue
            fi
            for seed in "${SEEDS[@]}"; do
              job_id=$((job_id + 1))
              TASK_MAX_LENGTH=$(echo "$split" | cut -d'_' -f2)

              job_file="evaluation_jobs/${split_strategy_prefix}/job_${job_id}_${split}_${pos_embedding_type}_${function_type}_${seed}_${eval_split}.sh"
              
              cat <<EOF > "$job_file"
#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=${split}_${pos_embedding_type}_${function_type}_${seed}_${eval_split}_${nsamples}
#SBATCH --output=./a_logs/evaluation/${split_strategy_prefix}/${split}_${pos_embedding_type}_${function_type}_${mode}_${seed}_${eval_split}_${nsamples}.out
#SBATCH --error=./a_logs/evaluation/${split_strategy_prefix}/${split}_${pos_embedding_type}_${function_type}_${mode}_${seed}_${eval_split}_${nsamples}.err

module load conda/latest
cd /scratch/workspace/ppruthi_umass_edu-MI/task_based_compositional_generalization
conda activate CG

python -m scripts.evaluate_model \\
  --prompt_mode "$mode" \\
  --train_split "$split" \\
  --eval_split "$eval_split" \\
  --nheads_nlayers "$NHEADS_NLAYERS" \\
  --pos_embedding_type "$pos_embedding_type" \\
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
done


echo "Generated $job_id job files in ./evaluation_jobs/${split_strategy_prefix}/"
