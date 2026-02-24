#!/bin/bash

# Parameters
PROMPT_LENGTHS=("fixed")
PROMPT_MODES=("step_by_step" "direct")
POS_EMBEDDING_TYPES=("abs" "rel_global")

# TRAIN_SPLIT_STRATEGIES=("reversecoverage_6_0_0" "reversecoverage_6_0_1" "reversecoverage_6_0_2" "reversecoverage_6_0_3" "reversecoverage_6_0_4" "reversecoverage_6_0_5" "reversecoverage_6_1_0" "reversecoverage_6_1_1" "reversecoverage_6_1_2" "reversecoverage_6_1_3" "reversecoverage_6_1_4" "reversecoverage_6_1_5" "reversecoverage_6_2_0" "reversecoverage_6_2_1" "reversecoverage_6_2_2" "reversecoverage_6_2_3" "reversecoverage_6_2_4" "reversecoverage_6_2_5" "reversecoverage_6_3_0" "reversecoverage_6_3_1" "reversecoverage_6_3_2" "reversecoverage_6_3_3" "reversecoverage_6_3_4" "reversecoverage_6_3_5" "reversecoverage_6_4_0" "reversecoverage_6_4_1" "reversecoverage_6_4_2" "reversecoverage_6_4_3" "reversecoverage_6_4_4" "reversecoverage_6_4_5" "reversecoverage_6_5_0" "reversecoverage_6_5_1" "reversecoverage_6_5_2" "reversecoverage_6_5_3" "reversecoverage_6_5_4" "reversecoverage_6_5_5")
TRAIN_SPLIT_STRATEGIES=("continuouspaircoverage_6_0.0" "continuouspaircoverage_6_0.1" "continuouspaircoverage_6_0.2" "continuouspaircoverage_6_0.3" "continuouspaircoverage_6_0.4" "continuouspaircoverage_6_0.5" "continuouspaircoverage_6_0.6" "continuouspaircoverage_6_0.7" "continuouspaircoverage_6_0.8" "continuouspaircoverage_6_0.9" "continuouspaircoverage_6_1.0")
# TRAIN_SPLIT_STRATEGIES=("systematiccontinuouscoverage_6_0.0" "systematiccontinuouscoverage_6_0.1" "systematiccontinuouscoverage_6_0.2" "systematiccontinuouscoverage_6_0.3" "systematiccontinuouscoverage_6_0.4" "systematiccontinuouscoverage_6_0.5" "systematiccontinuouscoverage_6_0.6" "systematiccontinuouscoverage_6_0.7" "systematiccontinuouscoverage_6_0.8" "systematiccontinuouscoverage_6_0.9" "systematiccontinuouscoverage_6_1.0" "randomcontinuouscoverage_6_0.0" "randomcontinuouscoverage_6_0.1" "randomcontinuouscoverage_6_0.2" "randomcontinuouscoverage_6_0.3" "randomcontinuouscoverage_6_0.4" "randomcontinuouscoverage_6_0.5" "randomcontinuouscoverage_6_0.6" "randomcontinuouscoverage_6_0.7" "randomcontinuouscoverage_6_0.8" "randomcontinuouscoverage_6_0.9" "randomcontinuouscoverage_6_1.0")
FUNCTION_TYPES=("diverse")
N_ALPHABETS=26
SEQ_LEN=6
N_FUNCTIONS=6
NHEADS_NLAYERS="nh6_nl3"
SEEDS=(0 10 20 30 40)
split_strategy_prefix="continuouspaircoverage_6_diverse_all"

mkdir -p evaluation_jobs/${split_strategy_prefix}/
mkdir -p a_logs/evaluation/${split_strategy_prefix}/

job_id=0
for split in "${TRAIN_SPLIT_STRATEGIES[@]}"; do
  for length in "${PROMPT_LENGTHS[@]}"; do
    for mode in "${PROMPT_MODES[@]}"; do
      for function_type in "${FUNCTION_TYPES[@]}"; do
        for pos_embedding_type in "${POS_EMBEDDING_TYPES[@]}"; do
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
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=${split}_${pos_embedding_type}_${function_type}_${seed}_${eval_split}
#SBATCH --output=./a_logs/evaluation/${split_strategy_prefix}/${split}_${pos_embedding_type}_${function_type}_${mode}_${seed}_${eval_split}.out
#SBATCH --error=./a_logs/evaluation/${split_strategy_prefix}/${split}_${pos_embedding_type}_${function_type}_${mode}_${seed}_${eval_split}.err

module load conda/latest
cd /scratch4/workspace/ppruthi_umass_edu-CG/task_based_compositional_generalization
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
EOF

              chmod +x "$job_file"
            done
          done
        done
      done
    done
  done
done

echo "Generated $job_id job files in ./evaluation_jobs/${split_strategy_prefix}/"
