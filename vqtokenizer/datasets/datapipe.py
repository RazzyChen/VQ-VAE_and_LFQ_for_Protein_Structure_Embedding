"""
vqtokenizer/datasets/datapipe.py
Description:
This module implements LMDB validation, MD5 checksum verification, and patch retrieval.
"""

import hashlib
import pickle
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

import lmdb


class ProteinStructureDataset(Dataset):
    def __init__(self, lmdb_folder: str):
        self.use_lmdb = False
        self.lmdb_env = None
        self.pdb_files = []
        lmdb_folder = Path(lmdb_folder)
        if lmdb_folder.exists():
            lmdb_files = list(lmdb_folder.glob("*.lmdb"))
            if lmdb_files:
                lmdb_path = lmdb_files[0]
                md5_path = lmdb_path.with_suffix(".lmdb.md5")
                if md5_path.exists():

                    def compute_md5(file_path):
                        hash_md5 = hashlib.md5()
                        with open(file_path, "rb") as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hash_md5.update(chunk)
                        return hash_md5.hexdigest()

                    with open(md5_path) as f:
                        saved_md5 = f.read().strip()
                    real_md5 = compute_md5(lmdb_path)
                    if saved_md5 == real_md5:
                        self.use_lmdb = True
                        self.lmdb_env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
                        with self.lmdb_env.begin() as txn:
                            self.pdb_files = [Path(key.decode()) for key, _ in txn.cursor()]
                        print(f"[LMDB] Loaded LMDB: {lmdb_path} (MD5 check passed, {len(self.pdb_files)} PDBs found)")
                        return
                    else:
                        print("[LMDB] MD5 check failed, ignoring LMDB.")
                else:
                    print(f"[LMDB] MD5 file not found: {md5_path}")
            else:
                print(f"[LMDB] No .lmdb file found in folder {lmdb_folder}")
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
            patches = torch.load(pickle.loads(pickle.dumps(tensor_bytes)), map_location="cpu")
        return patches


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
        self.dataset = ProteinStructureDataset(data_dir)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
