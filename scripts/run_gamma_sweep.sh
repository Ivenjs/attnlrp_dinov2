#!/bin/bash
#SBATCH --job-name=gamma_sweep
#SBATCH --chdir=/sc/home/iven.schlegelmilch/bachelor_thesis_code
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=6
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH -p aisc 
#SBATCH --account=aisc 
#SBATCH --qos=aisc 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slack:iven.schlegelmilch
#SBATCH --output=logs/%x-%j.out    # %x = jobname, %j = jobid
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL

# The first command-line argument is the EXPERIMENT NAME
# Any subsequent arguments will be treated as config overrides
EXPERIMENT_NAME=${1}

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Error: No experiment name provided."
    echo "Usage: sbatch $0 <experiment_name> [optional.key=value ...]"
    exit 1
fi

CONFIG_OVERRIDES="${@:2}"

echo "Starting job for experiment: ${EXPERIMENT_NAME}"
echo "With config overrides: ${CONFIG_OVERRIDES}"

# Container Settings
srun --container-image=/sc/home/iven.schlegelmilch/ivenschlegelmilch+gorillawatch+1.2.1.sqsh \
     --container-mount-home \
     --container-name=gorillawatch \
     --container-workdir=/workspaces \
     --container-mounts=/sc/home/iven.schlegelmilch/bachelor_thesis_code:/workspaces/bachelor_thesis_code,/sc/projects/sci-aisc/gorilla/:/workspaces/vast-gorilla \
     --container-writable \
     bash -c "cd /workspaces/bachelor_thesis_code && \
              /opt/conda/envs/research/bin/python src/bachelor_thesis/run_dinov2_attnlrp_sweep.py \
              --config_name ${EXPERIMENT_NAME} \
              ${CONFIG_OVERRIDES}"