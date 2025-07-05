# Define your paths first for clarity
CONTAINER_IMG="/home/iven.schlegelmilch/ivenschlegelmilch+gorillawatch+1.2.1.sqsh"
CODE_MOUNT="/home/iven.schlegelmilch/bachelor_thesis_code:/workspaces/bachelor_thesis_code"
DATA_MOUNT="/mnt/vast-gorilla:/workspaces/vast-gorilla"

SETUP_SCRIPT_PATH="/workspaces/bachelor_thesis_code/scripts/interactive_setup.sh"

# The command itself
srun --nodes=1 --ntasks=1 --gpus=1 --time=01:00:00 --partition=defq \
--container-image="$CONTAINER_IMG" \
--container-workdir=/workspaces \
--container-mounts="$CODE_MOUNT,$DATA_MOUNT" \
--export=ALL --pty \
bash 