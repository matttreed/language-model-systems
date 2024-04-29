#!/bin/bash
#SBATCH --job-name=distributed_nodes
#SBATCH --partition=a2
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=3
#SBATCH --nodes=2
#SBATCH --mem=8G
#SBATCH --time=00:02:00
#SBATCH --gpus-per-node=3
#SBATCH --error=scripts/logs/nodes.err
#SBATCH --output=scripts/logs/woha.err

LOG_FILE="scripts/logs/nodes.out"

exec >> ${LOG_FILE}

# Optional: activate a conda environment to use for this job
eval "$(conda shell.bash hook)"
conda activate cs336_systems

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# echo "MASTER_PORT: ${MASTER_PORT}"
# echo "MASTER_ADDR: ${MASTER_ADDR}"

export DEVICE=$1
export BACKEND=$2
export LEN_TENSOR=$3

echo -n "${LEN_TENSOR} ${DEVICE} ${BACKEND} "
srun python multinode_distributed.py
