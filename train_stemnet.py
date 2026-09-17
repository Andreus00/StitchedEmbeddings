import hydra
import lightning as pl
import torch
torch.set_float32_matmul_precision("medium")  # fast, safe default
import os 

@hydra.main(config_path="src/exp_configs", config_name="train_stemnet.yaml", version_base=None)
def main(cfg):
    if cfg.debug:
        cfg.trainer.callbacks[0]["dirpath"] = cfg.trainer.callbacks[0]["dirpath"] + "_" + str(cfg.num_elements_debug)
    if cfg.trainer.devices is None:
        if cfg.device == "cuda":
            num_devices = torch.cuda.device_count()
            cfg.trainer.devices = num_devices
        elif cfg.device == "cpu":
            cfg.trainer.devices = 1
    pl.seed_everything(cfg.seed)

    trainer = hydra.utils.instantiate(cfg.trainer)
    pl_model = hydra.utils.instantiate(cfg.model)
    pl_model.set_optimizer(cfg.optimizer)
    if hasattr(trainer.logger, "log_hyperparams"):
        trainer.logger.log_hyperparams(cfg)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    # pl_model.model.set_panel_edge_type_indices(datamodule.panel_edge_type_indices)


    # save the paths of the garments (in cfg.out_folder) to a file
    train_dataset_garment_names = datamodule.train_dataset.get_garment_names()
    os.makedirs(cfg.out_folder, exist_ok=True)
    out_file = os.path.join(cfg.out_folder, "garment_paths_train_dataset.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for name in train_dataset_garment_names:
            path = name
            f.write(path + "\n")
    print(f"Saved dataset of {len(train_dataset_garment_names)} garment paths to {out_file}")
    val_dataset_garment_names = datamodule.val_dataset.get_garment_names()
    os.makedirs(cfg.out_folder, exist_ok=True)
    out_file = os.path.join(cfg.out_folder, "garment_paths_val_dataset.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for name in val_dataset_garment_names:
            path = name
            f.write(path + "\n")
    print(f"Saved dataset of {len(val_dataset_garment_names)} garment paths to {out_file}")


    if cfg.model_ckpt is not None and isinstance(cfg.model_ckpt, str):
        trainer.fit(pl_model, datamodule=datamodule, ckpt_path=cfg.model_ckpt)
    else:
        trainer.fit(pl_model, datamodule=datamodule)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()