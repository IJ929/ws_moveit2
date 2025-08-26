#!/bin/bash
set -e

echo "📦 Running post-create commands..."

source /root/.bashrc
apt install -y python3-pip
pip install -r requirements.txt

# # --- 1. Set up Conda paths ---
# # The target for your mount is /root/conda/pkgs.
# # Therefore, the Conda installation directory should be one level up.
# CONDA_DIR="/root/conda"
# export PATH="${CONDA_DIR}/bin:$PATH"

# # --- 3. Install Miniforge ---
# echo "⚙️ Downloading and installing Miniforge..."
# curl -L -o miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
# bash miniforge.sh -b -u -p "${CONDA_DIR}"
# rm miniforge.sh

# # --- 4. Configure Conda to use the shared cache ---
# # This is the key step. Create a .condarc file that tells Conda
# # exactly where the package directory is. This path matches the `target` in your devcontainer.json.
# echo "pkgs_dirs:
#   - ${CONDA_DIR}/pkgs" > /root/.condarc

# # --- 5. Initialize Shell and Create Environment ---
# echo "Initializing Conda for your shell..."
# conda init bash

# echo "Creating Conda environment from environment.yml..."
# # When this runs, Conda will check the mounted pkgs_dirs first.
# # If packages exist, it will link them instead of re-downloading.
# # conda env create -f environment.yml
# # echo "conda activate ros_env" >> ./.bashrc

# echo "✅ Post-create script finished successfully."