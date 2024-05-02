#!/bin/bash
#SBATCH --job-name=test_training
#SBATCH --partition=a2
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --mem=20G
#SBATCH --time=00:04:00
#SBATCH --gpus-per-node=1
#SBATCH --error=scripts/logs/time_ddp.err
#SBATCH --output=scripts/logs/woah.out

LOG_FILE="scripts/logs/time_ddp.out"

exec >> ${LOG_FILE}

eval "$(conda shell.bash hook)"
conda activate cs336_systems

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# echo "MASTER_PORT: ${MASTER_PORT}"
# echo "MASTER_ADDR: ${MASTER_ADDR}"

srun python cs336-systems/cs336_systems/time_ddp.py --type $1 --version $2
