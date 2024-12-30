#!/bin/bash
#SBATCH --job-name=mace_multinode   # Job name
#SBATCH --nodes=2                   # Number of nodes
#SBATCH --ntasks-per-node=4         # Tasks per node (1 per GPU)
#SBATCH --gres=gpu:4                # Number of GPUs per node
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --time=24:00:00             # Time limit
#SBATCH --output=slurm-%j.out       # Standard output log
#SBATCH --error=slurm-%j.err        # Error log

# Set environment variables for SLURM-based distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Run the Accelerate script
accelerate launch \
    --config_file mace-alex-mptraj-multinode.yaml \
    --main_process_ip $MASTER_ADDR \
    --machine_rank    $SLURM_NODEID \
    --num_machines    $SLURM_NNODES \
    --num_processes   $SLURM_TASKS_PER_NODE \
    mace-alex-mptraj.py
