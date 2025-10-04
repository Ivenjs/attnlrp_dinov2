#!/bin/bash
#SBATCH --job-name=lrp_masking_analysis
#SBATCH --chdir=/sc/home/iven.schlegelmilch/attnlrp_dinov2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
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
     --container-mounts=/sc/home/iven.schlegelmilch/attnlrp_dinov2:/workspaces/attnlrp_dinov2,/sc/projects/sci-aisc/gorilla/:/workspaces/vast-gorilla \
     --container-writable \
     bash -c "cd /workspaces/attnlrp_dinov2 && \
              /opt/conda/envs/research/bin/python src/bachelor_thesis/run_mask_analysis.py \
              --config_name ${EXPERIMENT_NAME} \
              ${CONFIG_OVERRIDES}"
