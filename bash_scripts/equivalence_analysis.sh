#!/bin/bash
#SBATCH  -t 16:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --cpus-per-task 4
#SBATCH --job-name=equivalence_analysis_combination
#SBATCH --output=./a_logs/equivalence_analysis_combination.out
#SBATCH --error=./a_logs/equivalence_analysis_combination.err
python -m src.data.equivalence_classes.composition_equivalence_classes 