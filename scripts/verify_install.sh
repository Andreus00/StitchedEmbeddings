#!/bin/bash
# End-to-end smoke test on the bundled 3-garment sample (needs a CUDA GPU, ~5 min):
#   1. data processing  -> data/sample/.../<name>_udf_data_0.01.npz
#   2. training         -> 1 epoch of StemNet (1 GPU, TensorBoard logger)
#   3. testing          -> delta TTO on the UDF for the 3 sample garments (20 steps)
# Run from the repository root:  bash scripts/verify_install.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== [1/3] UDF precompute on the sample"
PYTHONPATH=. python scripts/prepare_data.py --root data/sample/GarmentCodeData_v2 --div 1 --part 0 --gpu 0
ls data/sample/GarmentCodeData_v2/garments_5000_0/default_body/data/*/*_udf_data_0.01.npz

echo "== [2/3] training smoke run (1 epoch, batch 1, single GPU; Hunyuan3D init weights used if present)"
INIT=checkpoints/hunyuan3d-vae-v2-1.fp16.ckpt; [ -f $INIT ] || INIT=null
HYDRA_FULL_ERROR=1 python train_stemnet.py hunyuan_init_ckpt=$INIT \
  trainer/logger=local_logger trainer.devices=1 trainer.strategy=auto \
  max_epochs=1 batch_size=1 accumulate_grad_batches=1 datamodule.num_workers=0 \
  out_folder=out_smoke trainer.callbacks.0.dirpath=out_smoke/checkpoints
ls out_smoke/checkpoints

echo "== [3/3] delta TTO on the sample test split (20 steps)"
python run_tto_delta_on_test_set.py --max_steps 20 --output results_tto_smoke
ls results_tto_smoke
echo "VERIFY_OK"
