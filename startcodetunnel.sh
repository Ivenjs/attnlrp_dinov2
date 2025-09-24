#!/bin/bash
set -e

# Use a unique directory specific to THIS project and THIS job
CLI_DATA_DIR="/workspaces/cadetunnel/bachelor_thesis_code/.vscode-cli-data-${SLURM_JOB_ID}"
mkdir -p "$CLI_DATA_DIR"

echo "Using VS Code CLI data directory: $CLI_DATA_DIR"

cd /workspaces/bachelor_thesis_code

if [ ! -f "code" ]; then
    echo "VS Code CLI not found. Downloading..."
    curl -Lk "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" --output vscode_cli.tar.gz
    tar -xf vscode_cli.tar.gz
    rm vscode_cli.tar.gz
fi

# Start the tunnel with its own unique data directory
./code tunnel --cli-data-dir "$CLI_DATA_DIR"
