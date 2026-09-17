#!/bin/bash
#SBATCH -J prepare-udf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=<INSER-HERE-YOUR-PARTITION>
#SBATCH --time=1-00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=./logs_slurm_data/prep_%A_%a.out
#SBATCH --error=./logs_slurm_data/prep_%A_%a.err
##SBATCH --mail-type=FAIL
##SBATCH --mail-user=<your-email>
set -euo pipefail

# UDF precompute: globs *_sim.{obj,ply} under ROOT, writes <name>_udf_data_0.01.npz
# next to each. Already-done (and valid) NPZs are skipped. --div = array size.
#   ROOT=../GarmentCode/garmentcodedata_v2/GarmentCodeData_v2 sbatch sbatch/run_prepare_data.sbatch
# `src.*` imports resolve via PYTHONPATH (submit from the repo root).
PYTHON=${PYTHON:-python}

mkdir -p ./logs_slurm_data

ROOT=${ROOT:-../GarmentCode/garmentcodedata_v2/GarmentCodeData_v2}
NSHARDS=8

echo "Precompute root=${ROOT}  shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS}"

PYTHONPATH=. "${PYTHON}" scripts/prepare_data.py \
  --root "${ROOT}" \
  --div "${NSHARDS}" \
  --part "${SLURM_ARRAY_TASK_ID}" \
  --gpu 0

echo "Shard ${SLURM_ARRAY_TASK_ID} done."
