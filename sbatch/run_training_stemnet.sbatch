#!/bin/bash

#SBATCH -J stemnet-train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=<INSER-HERE-YOUR-PARTITION>
#SBATCH --time=3-00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --output=./logs_slurm/stemnet_train_%j.out
#SBATCH --error=./logs_slurm/stemnet_train_%j.err
##SBATCH --mail-type=FAIL,END
##SBATCH --mail-user=<your-email>

set -euo pipefail

# ---------------------------------------------------------------------
# Job Information
# ---------------------------------------------------------------------
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID:              ${SLURM_JOB_ID}"
echo "Job Name:            ${SLURM_JOB_NAME}"
echo "Node:                ${SLURM_NODELIST}"
echo "Partition:           ${SLURM_JOB_PARTITION}"
echo "CPUs per task:       ${SLURM_CPUS_PER_TASK}"
echo "GPUs:                4 x A100"
echo "Memory:              64GB"
echo "Time limit:          3 days"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------
echo "=========================================="
echo "Final Model Configuration"
echo "=========================================="
echo "Config:              train_stemnet"
echo "Model:               StemNet"
echo "Prediction space:    discriminative_latents"
echo "Parameter predictor: PanelGroupedAttentionHeads"
echo "Head version:        v3"
echo "Attention heads:     8"
echo ""
echo "Loss Configuration:"
echo "  - KL loss:         1e-4"
echo "  - KL latent loss:  1e-2"
echo "  - UDF loss:        1.0"
echo "  - Param loss:      0.1"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------
echo "Setting up environment..."
# source ~/.bashrc
# conda activate stitched-final

# ---------------------------------------------------------------------
# Create log directory
# ---------------------------------------------------------------------
mkdir -p ./logs_slurm

# ---------------------------------------------------------------------
# Run Training
# ---------------------------------------------------------------------
echo "Starting training..."
echo ""

# Full dataset: GarmentCodeData v2 with the *_udf_data_0.01.npz produced by scripts/prepare_data.py
GCD_ROOT=${GCD_ROOT:-../GarmentCode/garmentcodedata_v2/GarmentCodeData_v2}
SPLIT=${SPLIT:-data/GarmentCodeData_v2_official_train_valid_test_data_split.json}

HYDRA_FULL_ERROR=1 \
python train_stemnet.py \
  --config-name=train_stemnet \
  root_dir=${GCD_ROOT} \
  train_val_test_file=${SPLIT} \
  hunyuan_init_ckpt=${HY3D_VAE_CKPT:-checkpoints/hunyuan3d-vae-v2-1.fp16.ckpt} \
  batch_size=16

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
