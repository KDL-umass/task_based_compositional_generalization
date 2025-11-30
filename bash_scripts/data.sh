#!/bin/bash
#SBATCH  -t 16:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --cpus-per-task 4
#SBATCH --job-name=data_test
#SBATCH --output=./a_logs/data_test.out
#SBATCH --error=./a_logs/data_test.err
module load conda/latest
cd /scratch4/workspace/ppruthi_umass_edu-CG/task_based_compositional_generalization
conda activate CG

# Generate data for within-k and cross-k evaluation with identity functions as task max length=7
# for prompt_length in "fixed"; do
#     echo "Generating data for prompt_length: $prompt_length"
#     for k in {1..6}; do
#         echo "Generating data for combination_$k"
#         python -m scripts.generate_data --prompt_length $prompt_length --split_strategy combination_$k --n_functions 6 --task_max_length 7 --n_alphabets 26 --seq_len 6 --functions_type uniform
#         python -m scripts.generate_data --prompt_length $prompt_length --split_strategy combination_$k --n_functions 6 --task_max_length 7 --n_alphabets 26 --seq_len 6 --functions_type diverse
        
#     done
# done

# # Generate data for within-k evaluation without identity functions (as task max length=k)
# for function_type in "uniform" "diverse"; do
#     echo "Generating data for function_type: $function_type"
#     for k in {2..6}; do
#         echo "Generating data for combination_$k"
        
#         python3 -m scripts.generate_data --split_strategy combination_$k --task_max_length $k --function_type $function_type
#     done
# done


# python3 -m scripts.generate_data --split_strategy combination_6 --task_max_length 6 --function_type diverse
# python3 -m scripts.generate_data --split_strategy disjoint_2_50 --task_max_length 2 --function_type diverse
# python3 -m scripts.generate_data --split_strategy disjoint_2_100 --task_max_length 2 --function_type diverse

# for percentage in 0 20 40 60 80 100; do
#     python3 -m scripts.generate_data --split_strategy disjoint1_4_$percentage --task_max_length 4 --function_type diverse
#     python3 -m scripts.generate_data --split_strategy disjoint3_4_$percentage --task_max_length 4 --function_type diverse
#     python3 -m scripts.generate_data --split_strategy disjoint4_4_$percentage --task_max_length 4 --function_type diverse
#     python3 -m scripts.generate_data --split_strategy disjoint5_4_$percentage --task_max_length 4 --function_type diverse
#     python3 -m scripts.generate_data --split_strategy disjoint6_4_$percentage --task_max_length 4 --function_type diverse
#     python3 -m scripts.generate_data --split_strategy disjoint7_4_$percentage --task_max_length 4 --function_type diverse
#     python3 -m scripts.generate_data --split_strategy disjoint8_4_$percentage --task_max_length 4 --function_type diverse
#     python3 -m scripts.generate_data --split_strategy disjoint9_4_$percentage --task_max_length 4 --function_type diverse
# done

python3 -m scripts.generate_data --split_strategy disjoint5_6_0 --task_max_length 6 --function_type diverse


# # Generate data for random sampling of compositions for K=6 for various train-test split ratios
# train_perm_size=(1 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700)
# for size in "${train_perm_size[@]}"; do
#     python -m scripts.generate_data --prompt_length fixed --split_strategy permutationrandom_6_$size --n_functions 6 --task_max_length 6 --n_alphabets 26 --seq_len 6 --functions_type uniform 
#     python -m scripts.generate_data --prompt_length fixed --split_strategy permutationrandom_6_$size --n_functions 6 --task_max_length 6 --n_alphabets 26 --seq_len 6 --functions_type diverse 
# done

# # generate data for systematic sampling of compositions for K=6 for various train-test split ratios
# train_perm_size=(1 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700)
# for size in "${train_perm_size[@]}"; do
#     python -m scripts.generate_data --prompt_length fixed --split_strategy permutation_6_$size --n_functions 6 --task_max_length 6 --n_alphabets 26 --seq_len 6 --functions_type uniform 
#     python -m scripts.generate_data --prompt_length fixed --split_strategy permutation_6_$size --n_functions 6 --task_max_length 6 --n_alphabets 26 --seq_len 6 --functions_type diverse 
# done

# # generate data to validate composition equivalence necessity using uniform-identity based equivalences
# # We focus on K=6, keep train perm size fixed as 576, test perm size fixed as 144 and systematically increase number of shared compositions in train and test
# shared_equivalence_size=(0 1 25 49 73 97 121 144)
# for size in "${shared_equivalence_size[@]}"; do
#     python -m scripts.generate_data --prompt_length fixed --split_strategy equivalence_6_576_$size --n_functions 6 --task_max_length 7 --n_alphabets 26 --seq_len 6 --functions_type uniform 
# done

# # generate data to validate number of training equivalences needed to learn corresponding test equivalences
# # We focus on K=6, keep train perm size fixed as 576, and shared equivalences as 100% and systematically increase number of training equivalences
# training_equivalence_size=(0)
# for size in "${training_equivalence_size[@]}"; do
#     python -m scripts.generate_data --prompt_length fixed --split_strategy uniequivalence_6_576_$size --n_functions 6 --task_max_length 7 --n_alphabets 26 --seq_len 6 --functions_type uniform 
# done

