#!/bin/bash
#SBATCH --job-name=vscode
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=scripts/logs/vscode_%j.out
#SBATCH --error=scripts/logs/vscode_%j.err
#SBATCH --mem=10G
#SBATCH --nodelist=ad12a3ca-01

sleep 12h