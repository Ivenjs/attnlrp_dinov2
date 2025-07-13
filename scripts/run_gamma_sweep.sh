#!/bin/bash
#SBATCH --job-name=gamma_sweep
#SBATCH --chdir=/sc/home/iven.schlegelmilch/bachelor_thesis_code
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH -w gx08,gx09,gx11,gx12,gx13
#SBATCH -p aisc 
#SBATCH --account=aisc 
#SBATCH --qos=aisc 
#SBATCH --output=logs/%x-%j.out    # %x = jobname, %j = jobid
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL

# Container Settings
# --container-mounts=/sc/home/iven.schlegelmilch/bachelor_thesis_code:/workspaces/bachelor_thesis_code,/mnt/vast-gorilla:/workspaces/vast-gorilla \

srun --container-image=/sc/home/iven.schlegelmilch/ivenschlegelmilch+gorillawatch+1.2.1.sqsh \
     --container-workdir=/workspaces \
     --container-mounts=/sc/home/iven.schlegelmilch/bachelor_thesis_code:/workspaces/bachelor_thesis_code\
     --container-writable \
     bash -c "cd /workspaces/bachelor_thesis_code/src/bachelor_thesis && /opt/conda/envs/research/bin/python run_dinov2_attnlrp_sweep.py"
