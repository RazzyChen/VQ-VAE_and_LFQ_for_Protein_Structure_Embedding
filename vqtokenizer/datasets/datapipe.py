"""
vqtokenizer/datasets/datapipe.py
Description:
This module implements LMDB validation, MD5 checksum verification, and patch retrieval.
"""

import hashlib
import io
import pickle
from pathlib import Path

import lmdb
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset


class ProteinStructureDataset(Dataset):
    def __init__(self, lmdb_folder: str):
        self.use_lmdb = False
        self.lmdb_env = None
        self.pdb_files = []
        lmdb_folder = Path(lmdb_folder)
        if lmdb_folder.exists():
            data_mdb = lmdb_folder / "data.mdb"
            md5_path = lmdb_folder / "data.md5"
            if data_mdb.exists() and md5_path.exists():

                def compute_md5(file_path):
                    hash_md5 = hashlib.md5()
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                    return hash_md5.hexdigest()

                with open(md5_path) as f:
                    saved_md5 = f.read().strip()
                real_md5 = compute_md5(data_mdb)
                if saved_md5 == real_md5:
                    self.use_lmdb = True
                    self.lmdb_env = lmdb.open(str(lmdb_folder), readonly=True, lock=False)
                    with self.lmdb_env.begin() as txn:
                        self.pdb_files = [Path(key.decode()) for key, _ in txn.cursor()]
                    print(f"[LMDB] Loaded LMDB: {data_mdb} (MD5 check passed, {len(self.pdb_files)} PDBs found)")
                    return
                else:
                    print("[LMDB] MD5 check failed, ignoring LMDB.")
            else:
                print(f"[LMDB] data.mdb or data.md5 not found in folder {lmdb_folder}")
        else:
            print(f"[LMDB] Folder {lmdb_folder} does not exist")

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if not self.use_lmdb or self.lmdb_env is None:
            raise RuntimeError("No valid LMDB file found. Please run the preprocessing script to generate LMDB first.")
        pdb_file = str(self.pdb_files[idx])
        with self.lmdb_env.begin() as txn:
            tensor_bytes = txn.get(pdb_file.encode())
            if tensor_bytes is None:
                raise KeyError(f"{pdb_file} not found in LMDB")
            # 用 io.BytesIO 包装 bytes 数据
            patches = torch.load(io.BytesIO(tensor_bytes), map_location="cpu")
        return patches


def pad_collate_fn(batch, patch_size):
    # batch: List[Tensor], each shape [num_patches, patch_size*4]
    max_num_patches = max(x.shape[0] for x in batch)
    feature_dim = patch_size * 4
    padded = []
    for x in batch:
        num_patches = x.shape[0]
        if num_patches < max_num_patches:
            pad_tensor = torch.zeros((max_num_patches - num_patches, feature_dim), dtype=x.dtype)
            padded_x = torch.cat([x, pad_tensor], dim=0)
        else:
            padded_x = x[:max_num_patches]
        padded.append(padded_x)
    return torch.stack(padded, dim=0)


class ProteinDataModule(pl.LightningDataModule):
    def __init__(
        self, pdb_dir: str, lmdb_dir: str = None, patch_size: int = 8, batch_size: int = 64, num_workers: int = 4
    ):
        super().__init__()
        self.pdb_dir = pdb_dir
        self.lmdb_dir = lmdb_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Use lmdb_dir if provided, else fallback to pdb_dir
        data_dir = self.lmdb_dir if self.lmdb_dir is not None else self.pdb_dir
        md5_path = Path(data_dir) / "data.md5"
        if not md5_path.exists():
            print(f"[ERROR] data.md5 not found at: {md5_path}")
            print(
                "Generally, the md5 and mdb files are located together, which means the data was not correctly recognized."
            )
            print("Abort training.")
            import sys

            sys.exit(1)
        self.dataset = ProteinStructureDataset(data_dir)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda batch: pad_collate_fn(batch, self.patch_size),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: pad_collate_fn(batch, self.patch_size),
        )
