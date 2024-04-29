#!/bin/bash
#SBATCH --job-name=benchmark_all_reduce
#SBATCH --partition=a2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --gpus=6
#SBATCH --output=scripts/logs/benchmark_all_reduce_%j.out
#SBATCH --error=scripts/logs/benchmark_all_reduce_%j.err
#SBATCH --mem=100G

# Optional: activate a conda environment to use for this job
eval "$(conda shell.bash hook)"
conda activate cs336_basics

python benchmark_all_reduce.py --device cpu --backend gloo