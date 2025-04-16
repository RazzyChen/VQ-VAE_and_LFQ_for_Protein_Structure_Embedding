from pathlib import Path

import yaml

config = {
    "wandb": {"project": "VQTokenizer"},
    "data": {"pdb_dir": "./data", "patch_size": 3, "batch_size": 64, "num_workers": 12},
    "model": {"hidden_dim": 512, "latent_dim": 256, "num_embeddings": 1024, "nhead": 8, "learning_rate": 1e-3},
    "trainer": {
        "max_epochs": 50,
        "accelerator": "auto",
        "checkpoint_dir": "./checkpoints",
        "checkpoint_name": "vqtokenizer-{epoch:02d}-{val_total_loss:.4f}",
        "save_top_k": 5,
        "monitor": "val_total_loss",
        "monitor_mode": "min",
        "log_every_n_steps": 100,
        "val_check_interval": 0.25,
        "final_model_path": "./weight/final_model.ckpt",
        "precision": "fp16",
        "strategy": "ddp",
    },
}

output_path = Path("config/train.yml")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    yaml.dump(config, f, sort_keys=False)

print(f"train.yml at {output_path.resolve()}")
