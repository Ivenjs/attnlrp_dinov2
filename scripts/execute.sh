#!/bin/bash -ex

# ===================================================================
# SLURM Job Configuration for Iven Schlegelmilch
# ===================================================================

# --- Job Details ---
#SBATCH --job-name=dinov2-attnlrp
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

# --- Logging ---
# Create a logs directory and save output there, organized by Job ID
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
mkdir -p logs

# --- Container Configuration ---
#SBATCH --container-image=/home/iven.schlegelmilch/ivenschlegelmilch+gorillawatch+1.2.1.sqsh
#SBATCH --container-workdir=/workspaces
#SBATCH --container-mounts=/home/iven.schlegelmilch/bachelor_thesis_code:/workspaces/bachelor_thesis_code,/mnt/vast-gorilla:/workspaces/vast-gorilla

# ===================================================================
# Environment Setup
# ===================================================================

# --- Set the Hugging Face Cache to a WRITABLE, PERSISTENT location ---
# We use the path on the HOST machine, as sbatch sets this before the job starts.
# --export=ALL will pass this variable into the container.
export HF_HOME="/mnt/vast-gorilla/.cache/huggingface"

# Create the directory beforehand to be safe
mkdir -p "$HF_HOME"

echo "==============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Hugging Face cache is set to: $HF_HOME"
echo "==============================================================="

# ===================================================================
# Execute the Python Script
# ===================================================================

# Use srun to launch the command inside the allocated resources.
# The --export=ALL flag ensures HF_HOME is available inside the container.
srun --export=ALL python /workspaces/bachelor_thesis_code/src/bachelor_thesis/run_dinov2_attnlrp.py

echo "==============================================================="
echo "Job finished with exit code $?"
echo "==============================================================="