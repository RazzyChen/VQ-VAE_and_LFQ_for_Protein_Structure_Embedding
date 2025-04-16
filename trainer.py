"""
trainer.py
功能说明：
本模块为 VQTokenizer 项目的训练脚本，负责读取配置、初始化 wandb、数据模块、模型、训练器并执行训练。
"""

import argparse
import os
import sys

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from vqtokenizer.datasets.datapipe import ProteinDataModule
from vqtokenizer.nn.backbone import VQTokenizer


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Initialize wandb
    wandb.init(project=cfg.wandb.project)

    # Create data module
    data_module = ProteinDataModule(
        pdb_dir=cfg.data.pdb_dir,
        patch_size=cfg.data.patch_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # Calculate input dimension
    input_dim = cfg.data.patch_size * 4  # patch_size * 2 angles * 2 (sin/cos)

    # Create model
    model = VQTokenizer(
        input_dim=input_dim,
        hidden_dim=cfg.model.hidden_dim,
        latent_dim=cfg.model.latent_dim,
        num_embeddings=cfg.model.num_embeddings,
        nhead=cfg.model.nhead,
        learning_rate=cfg.model.learning_rate,
    )

    # Create wandb logger
    wandb_logger = WandbLogger(project=cfg.wandb.project)

    # Create model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.trainer.checkpoint_dir,
        filename=cfg.trainer.checkpoint_name,
        save_top_k=cfg.trainer.save_top_k,
        monitor=cfg.trainer.monitor,
        mode=cfg.trainer.monitor_mode,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        precision=cfg.trainer.precision,
    )

    # Train model
    trainer.fit(model, data_module)

    # Save final model
    trainer.save_checkpoint(cfg.trainer.final_model_path)

    # Finish wandb
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQTokenizer Trainer: Training for protein structure VQ models.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to config file (e.g. config/train.yml). If not specified, use hydra mode.",
    )
    args, unknown = parser.parse_known_args()

    if args.config:
        if not os.path.exists(args.config):
            print(f"[ERROR] Config file {args.config} does not exist!")
            sys.exit(1)
        cfg = OmegaConf.load(args.config)
        main(cfg)
    else:
        config_path = os.path.join(os.path.dirname(__file__), "config", "train.yml")
        if len(sys.argv) == 1:
            if not os.path.exists(config_path):
                print(
                    "[ERROR] config/train.yml does not exist. Please run create_default_yaml.py or create the config file manually!"
                )
                sys.exit(1)
            print("[ERROR] Please run with:\n  python trainer.py --config-path config --config-name train.yml")
            sys.exit(1)
        else:
            main(OmegaConf.load(config_path))
