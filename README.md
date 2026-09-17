# Stitched Embeddings — train + test

<p align="center">
    <img src="https://andreus00.github.io/stitchedembeddings/static/images/teaser_final.png" alt="Stitched Embeddings teaser" width="65%">
</p>
<p align="center">
    <img src="https://andreus00.github.io/assets/logos/ECCV_2026_Logo.svg" alt="ECCV 2026" width="20%">
</p>

<h3 align="center">
    <a href='https://arxiv.org/abs/2607.00829'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp;
    <a href='https://andreus00.github.io/stitchedembeddings/'><img src='https://img.shields.io/badge/Project-Page-blue'></a> &nbsp;
</h3>

Official implementation of *Stitched Embeddings: A Unified Latent Space for 3D Garments and 2D Patterns*.
The repository contains the minimal code to **train** StemNet (the `StemNet` VAE with
`PanelGroupedAttentionHeads`, trained with the KL, KL-latent, UDF and design-parameter losses) on
GarmentCodeData v2, and to **test** the released checkpoint with **delta test-time optimisation on the
UDF** (`run_tto_delta_on_test_set.py`).

Everything needed to run is inside this folder except the full dataset:

```
train_stemnet.py            Hydra entry point — training (W1)
run_tto_delta_on_test_set.py   argparse entry point — delta TTO + pattern metrics on a test split (W2)
models.py                      builds the model and loads checkpoints/best-checkpoint-v26.ckpt
evaluate_specification.py      pattern metrics (panel matching, vertex L2, stitch precision, ...)
scripts/prepare_data.py        data processing: sim + boxmesh meshes -> <name>_udf_data_0.01.npz
scripts/verify_install.sh      end-to-end test on the bundled sample
sbatch/                        SLURM scripts: data preparation, training, TTO test (1 GPU)
src/model/                     training_loop_lightning.TrainingLoopLightning (LightningModule), vae/ (encoders, VAE, heads)
src/datamodule/                GCD_ALL dataset + augmentation + LightningDataModule
src/util/                      design-parameter reader / heads, KL loss, UDF sampling, collate
src/exp_configs/               Hydra configs (train_stemnet.yaml + groups)
src/external/                  bundled third-party code: GarmentCode (unmodified), Hunyuan3D-2.1 VAE architecture, MeshUDF (modified)
data/sample/                   3 official-test garments (pants, dress, shirt) + split.json
data/neutral_body/             body mesh + measurements of the dataset's neutral body
data/GarmentCodeData_v2_official_train_valid_test_data_split.json   official split (full dataset)
checkpoints/best-checkpoint-v26.ckpt                 reference StemNet checkpoint (not in git, see "Checkpoints") — the only file needed for testing
checkpoints/hunyuan3d-vae-v2-1.fp16.ckpt             (training only, NOT included) original Hunyuan3D-2.1 shape-VAE weights
                                                     that initialise the encoders / UDF decoder — see "Training"
```

## Checkpoints

Download the weights from the release folder
**https://drive.google.com/drive/folders/1mxNBUEOibmkVhDkvbmyi0XGq2vNA1q04?usp=sharing** and place them in `checkpoints/`:

| file | size | use |
|---|---|---|
| `best-checkpoint-v26.ckpt` | 1.9 GB | StemNet weights for testing / fine-tuning (md5 `5f36efdd8ceeebb57a6b0ac9a841c3ff`) |
| `best-checkpoint-v26_full.ckpt` (optional) | 5.7 GB | the same weights plus the Lightning optimizer state, only needed to *resume* the original training run with `model_ckpt=` (md5 `05d621cd66d56b66a8a1585908f8ad49`) |

## Installation

Tested on Ubuntu 22.04, NVIDIA driver 565 (CUDA 12.x), Python 3.9, PyTorch 2.5.1 + CUDA 12.1.
Python **3.9 is required** to use the prebuilt MeshUDF marching-cubes extension
(`src/external/MeshUDF/custom_mc/*.cpython-39*.so`, used to extract validation meshes);
on another Python rebuild it with `cd src/external/MeshUDF/custom_mc && python setup.py build_ext --inplace`
(needs Cython).

```bash
conda create -n stitched-final python=3.9 -y
conda activate stitched-final

# 1. PyTorch 2.5.1 + CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 2. CUDA geometry extensions, prebuilt for torch 2.5.1 / cu121
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
pip install torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# 3. Everything else (also covers the bundled GarmentCode / Hunyuan3D / MeshUDF code)
pip install -r requirements.txt
```

`cairosvg` (pattern PNG export) needs the system `libcairo2`; if `python -c "import cairosvg"` fails, run
`conda install -c conda-forge cairo` (or `apt install libcairo2`).

Verify the installation (GPU required, ~5 minutes; runs data processing, one training epoch and a
short TTO on the bundled sample):

```bash
bash scripts/verify_install.sh        # ends with VERIFY_OK
```

Verified on a clean env built exactly as above (RTX 4090, 47 s): the three sample NPZs are produced, one
training epoch runs (losses logged, `best-checkpoint.ckpt` + `last.ckpt` written), and the 20-step TTO
reports 6/6, 6/6 and 12/12 matched panels for the three garments (`summary.json`). Nothing is fetched from
the network at run time. Testing needs only the StemNet checkpoint: the Hunyuan3D architecture is built
from the bundled hyper-parameter file `src/external/Hunyuan3D/configs/hunyuan3d-vae-v2-1.yaml`.

Weights & Biases is the default training logger (`wandb login` first, or add
`trainer/logger=local_logger` to log to TensorBoard under `logs/`).

## Third-party code (`src/external/`)

`GarmentCode/` is an unmodified subset of [GarmentCode](https://github.com/maria-korosteleva/GarmentCode) @ `d449629`;
`Hunyuan3D/` (Hunyuan3D-2.1 shape-VAE architecture code + StemNet's `*_v2` boundary-aware encoder) and
`MeshUDF/custom_mc/` (marching cubes with the `th_alpha` / `th_beta` thresholds) are modified copies that the model
needs — nothing else has to be cloned or installed for them; their Python dependencies are covered by
`requirements.txt`. No Tencent model weights are distributed.

## Data

Full dataset (not included): GarmentCodeData v2, expected as
`../GarmentCode/garmentcodedata_v2/GarmentCodeData_v2/garments_5000_X/default_body/data/<name>/`
with `<name>_sim.ply`, `<name>_boxmesh.ply`, `<name>_design_params.yaml`, `<name>_specification.json`.
Any other location works through `root_dir=` / `--data_root`.

**Data processing** (once per dataset root) writes the training file `<name>_udf_data_0.01.npz`
(surface points of garment and boxmesh, UDF query points and values) next to each garment:

```bash
PYTHONPATH=. python scripts/prepare_data.py --root <dataset root> --div 1 --part 0 --gpu 0
# or sharded on SLURM (array job, 8 shards):
# note: remember to set your partition in the sbatch file
ROOT=<dataset root> sbatch sbatch/run_prepare_data.sbatch
```

## Training

```bash
# bundled sample (defaults in src/exp_configs/train_stemnet.yaml)
python train_stemnet.py

# full dataset, reference settings (4 x A100, batch 16, grad-accumulation 2, 200 epochs)
# note: remember to set your partition in the sbatch file
sbatch sbatch/run_training_stemnet.sbatch
#  = python train_stemnet.py root_dir=<dataset root> \
#        train_val_test_file=data/GarmentCodeData_v2_official_train_valid_test_data_split.json batch_size=16
```

Any config key can be overridden on the command line (`lr=5e-5`, `max_epochs=100`, `kl_lambda=1e-4`,
`trainer.devices=1`, ...). Checkpoints go to `checkpoints/stemnet_training/`
(`best-checkpoint.ckpt` on `avg_val_ckpt_loss`, plus `last.ckpt`); `model_ckpt=<path>` resumes.

**Initialisation.** The encoders and the UDF decoder are initialised from the original Hunyuan3D-2.1
shape-VAE weights, which are not bundled: download `hunyuan3d-vae-v2-1/model.fp16.ckpt` from
[tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) (subject to the Tencent Hunyuan 3D 2.1
Community License) and save it as `checkpoints/hunyuan3d-vae-v2-1.fp16.ckpt` (config key `hunyuan_init_ckpt`;
the sbatch script reads `$HY3D_VAE_CKPT`). `hunyuan_init_ckpt=null` trains from random weights (smoke tests only);
a fine-tuned Hunyuan3D checkpoint (`{'model': state_dict}`) is accepted too — the reference checkpoint v26 was
trained from such a fine-tune. Inference never needs these weights: the StemNet checkpoint holds all of them.

Losses (weights in the config): `kl_lambda` 1e-4, `kl_latent_lambda` 1e-2, `udf_lambda` 1,
`param_lambda` 0.1 (masked design-parameter loss). The RDM loss of earlier experiments is not part
of this model.

## Testing — delta TTO on the UDF

For every test garment: encode the simulated mesh, read the pattern (initial prediction), then
optimise a delta on the latent against the garment's UDF (`loss = udf + delta_reg * mean(delta^2) +
kl_reg * 0.5 * mean(z^2)`) and read the pattern again (final prediction). Both predictions are
scored against the ground-truth specification.

```bash
# bundled sample
python run_tto_delta_on_test_set.py --max_steps 300 --output results_tto_sample

# official test split of the full dataset (reference settings)
sbatch sbatch/run_tto_delta_on_test_set.sbatch
#  = python run_tto_delta_on_test_set.py --max_steps 1000 --lr 1e-3 --delta_reg_weight 1.0 --kl_reg_weight 1e-3 \
#        --ckpt checkpoints/best-checkpoint-v26.ckpt --data_root <dataset root> \
#        --split_file data/GarmentCodeData_v2_official_train_valid_test_data_split.json --output results_tto_delta_testset
```

Per garment the output folder holds `gt_pattern.json`, `sim_gt.ply`, `initial_prediction/`,
`final_prediction/` (specification + SVG/PNG of the pattern), `optimization_metrics.csv`, and
`summary.json` aggregates the metrics over the split. `--max_folders N` limits the number of garments,
`--eval_every N` scores intermediate steps.

## Acknowledgements

This code builds on [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1),
[MeshUDF](https://github.com/cvlab-epfl/MeshUDF) and [GarmentCode](https://github.com/maria-korosteleva/GarmentCode) —
thanks to their authors for releasing them.

## Citation

```bibtex
@inproceedings{sanchietti2026stitched,
  title={Stitched Embeddings: A Unified Latent Space for 3D Garments and 2D Patterns},
  author={Sanchietti, Andrea and Marin, Riccardo and Bhatnagar, Bharat Lal and Xu, Yuanlu and Pons-Moll, Gerard},
  booktitle={European Conference on Computer Vision},
  pages={235--255},
  year={2026},
  organization={Springer}
}
```
