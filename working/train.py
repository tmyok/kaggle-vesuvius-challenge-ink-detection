
import hydra
import math
import torch
import wandb

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from dataset import MyDataModule
from model import Unet3D, SegFormer

@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):

    if cfg.debug:
        cfg.model.num_epochs = 1
        cfg.wandb_logger = False

    job_type = cfg.model.type

    output_dir = Path(f"../output/{job_type}/fold{cfg.fold}")
    output_log_dir = Path(f"../output/wandb/{job_type}/fold{cfg.fold}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_log_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed=cfg.random_seed, workers=True)
    torch.set_float32_matmul_precision('high')
    device_count = torch.cuda.device_count()

    # dataset
    data_module = MyDataModule(cfg)
    data_module.setup()

    # model
    if cfg.model.type == "mit_b2":
        model = SegFormer(
            cfg=cfg,
            steps_per_epoch=math.ceil(len(data_module.train_dataloader())/device_count),
        )
    else:
        model = Unet3D(
            cfg=cfg,
            steps_per_epoch=math.ceil(len(data_module.train_dataloader())/device_count),
        )

    # logger
    if cfg.wandb_logger:
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            job_type=job_type,
            name=f"fold{cfg.fold}",
            log_model=False,
            settings=wandb.Settings(start_method="fork"),
            save_dir=output_log_dir,
        )
        wandb_logger.log_hyperparams(cfg)
    else:
        wandb_logger = None

    callbacks = []
    callbacks.append(ModelCheckpoint(
        dirpath=output_dir,
        monitor="val_dice",
        filename="val_dice",
        save_weights_only=True,
        save_top_k=1,
        mode="max",))

    trainer = Trainer(
        # env
        deterministic="warn",
        accelerator="gpu",
        devices=device_count,
        precision="16-mixed" if cfg.model.type == "mit_b2" else "bf16-mixed",
        # training
        fast_dev_run=cfg.debug,
        enable_model_summary=True,
        max_epochs=cfg.model.num_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    trainer.fit(model, data_module)

    if cfg.wandb_logger:
        wandb.finish()

    val_dice = trainer.callback_metrics["val_dice"].item()

    if trainer.global_rank == 0:
        print(f"val_dice={val_dice}")

    return val_dice

if __name__ == '__main__':
    main()