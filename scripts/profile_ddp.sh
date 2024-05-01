#!/bin/bash
#SBATCH --job-name=test_training
#SBATCH --partition=a2
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2
#SBATCH --mem=20G
#SBATCH --time=00:02:00
#SBATCH --gpus-per-node=2
#SBATCH --error=scripts/logs/profile_ddp.err
#SBATCH --output=scripts/logs/profile_ddp.out


eval "$(conda shell.bash hook)"
conda activate cs336_systems

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# echo "MASTER_PORT: ${MASTER_PORT}"
# echo "MASTER_ADDR: ${MASTER_ADDR}"

srun python cs336-systems/cs336_systems/profile_ddp.py --version $2 --type $1
