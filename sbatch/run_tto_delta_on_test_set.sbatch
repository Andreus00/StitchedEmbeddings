#!/bin/bash

#SBATCH -J tto-delta-testset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=<INSER-HERE-YOUR-PARTITION>
#SBATCH --time=1-00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs_slurm_hparam/tto_delta_%j.out
#SBATCH --error=./logs_slurm_hparam/tto_delta_%j.err
##SBATCH --mail-type=FAIL,END
##SBATCH --mail-user=<your-email>

set -euo pipefail

mkdir -p logs_slurm_hparam

echo "Node: ${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi
ECC_ERRORS=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits | tr -d ' ')
if [ "$ECC_ERRORS" -gt 0 ] 2>/dev/null; then
    echo "ERROR: GPU has $ECC_ERRORS uncorrectable ECC errors — aborting on bad node ${SLURM_NODELIST}"
    exit 1
fi

GCD_ROOT=${GCD_ROOT:-../GarmentCode/garmentcodedata_v2/GarmentCodeData_v2}
SPLIT=${SPLIT:-data/GarmentCodeData_v2_official_train_valid_test_data_split.json}

python run_tto_delta_on_test_set.py \
    --max_steps 1000 \
    --lr 1e-3 \
    --delta_reg_weight 1.0 \
    --kl_reg_weight 1e-3 \
    --output results_tto_delta_testset \
    --ckpt checkpoints/best-checkpoint-v26.ckpt \
    --data_root ${GCD_ROOT} \
    --split_file ${SPLIT}
