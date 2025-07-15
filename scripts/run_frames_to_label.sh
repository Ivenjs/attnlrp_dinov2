#!/bin/bash
#SBATCH --job-name=extract_frames
#SBATCH --chdir=/sc/home/iven.schlegelmilch/bachelor_thesis_code
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --exclude=ga03
#SBATCH -p aisc 
#SBATCH --account=aisc 
#SBATCH --qos=aisc 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slack:iven.schlegelmilch
#SBATCH --output=logs/%x-%j.out    # %x = jobname, %j = jobid
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL

# Container Settings

srun --container-image=/sc/home/iven.schlegelmilch/ivenschlegelmilch+gorillawatch+1.2.1.sqsh \
     --container-name=gorillawatch \
     --container-workdir=/workspaces \
     --container-mounts=/sc/home/iven.schlegelmilch/bachelor_thesis_code:/workspaces/bachelor_thesis_code,/sc/projects/sci-aisc/gorilla/:/workspaces/vast-gorilla \
     --container-writable \
     bash -c "cd /workspaces/bachelor_thesis_code/src/bachelor_thesis && /opt/conda/envs/research/bin/python get_frames_to_label.py"
