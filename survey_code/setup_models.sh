#!/bin/bash
set -euo pipefail

# ==========================================
# HPC MODEL AND DATASET SETUP TEMPLATE
# ==========================================

# Project directory
PROJECT_DIR="${PROJECT_DIR:-$HOME/autonomous_driving}"
MODEL_DIR="${PROJECT_DIR}/models"
DATASET_DIR="${PROJECT_DIR}/datasets"

mkdir -p "$MODEL_DIR" "$DATASET_DIR"

echo "Project directory: $PROJECT_DIR"
echo "Model directory:   $MODEL_DIR"
echo "Dataset directory: $DATASET_DIR"

# Activate your environment if required
# module load python
# module load cuda
# source ~/venvs/vlm/bin/activate

# Install download utility if not already installed
python -m pip install -U huggingface_hub

# ==========================================
# MODEL REPOSITORIES
# Exact checkpoints used in the paper (Section VII).
# ==========================================

QWEN_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
LLAVA_MODEL="llava-hf/llava-1.5-7b-hf"
INTERNVL_MODEL="OpenGVLab/InternVL3-8B-hf"

# Download models only when enabled
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-false}"

if [[ "$DOWNLOAD_MODELS" == "true" ]]; then
    huggingface-cli download "$QWEN_MODEL" \
        --local-dir "$MODEL_DIR/qwen"

    huggingface-cli download "$LLAVA_MODEL" \
        --local-dir "$MODEL_DIR/llava"

    huggingface-cli download "$INTERNVL_MODEL" \
        --local-dir "$MODEL_DIR/internvl"
else
    echo "Model downloads skipped."
    echo "Set DOWNLOAD_MODELS=true to enable downloads."
fi

# ==========================================
# DATASET
# Download DriveLM-nuScenes separately using
# the official dataset instructions.
# ==========================================

echo "Dataset directory: $DATASET_DIR/drivelm_nuscenes"
mkdir -p "$DATASET_DIR/drivelm_nuscenes"

# Add your approved dataset download commands here.
# Do not assume the dataset is a single direct URL.

echo "Setup script completed."
