# This script will be sourced by bash when our interactive session starts.

# Announce that the custom setup is running
echo ">>> Sourcing custom setup file: interactive_setup.sh"

# --- Define and export cache directories ---
# Path is what the container will see
CACHE_DIR="/workspaces/bachelor_thesis_code/.cache"
export HF_HOME="${CACHE_DIR}/huggingface"
export XDG_CACHE_HOME="${CACHE_DIR}"
mkdir -p "$HF_HOME"
echo ">>> Cache directory set to: $HF_HOME"

# --- Activate your conda environment automatically ---
echo ">>> Activating conda environment: research"
source /opt/conda/etc/profile.d/conda.sh
conda activate research

# Optional: Add a nice indicator to your prompt
PS1="\[\e[32m\](research) \[\e[m\]${PS1}"

echo "------------------------------------------------------------------"
echo ">>> Environment is ready. Welcome to your interactive session."