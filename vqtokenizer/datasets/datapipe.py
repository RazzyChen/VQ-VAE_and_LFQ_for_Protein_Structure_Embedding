"""
vqtokenizer/datasets/datapipe.py
Description:
This module implements LMDB validation, MD5 checksum verification, and patch retrieval.
"""

import hashlib
import pickle
from pathlib import Path

import lmdb
import torch
from torch.utils.data import Dataset


class ProteinStructureDataset(Dataset):
    """
    只负责LMDB判定、MD5校验和读取patches。
    """

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
                        print(f"[LMDB] 加载LMDB: {lmdb_path} (MD5校验通过, 共{len(self.pdb_files)}个PDB)")
                        return
                    else:
                        print("[LMDB] MD5校验失败，忽略LMDB。")
                else:
                    print(f"[LMDB] 未找到MD5文件: {md5_path}")
            else:
                print(f"[LMDB] 在文件夹 {lmdb_folder} 中未找到.lmdb文件")
        else:
            print(f"[LMDB] 文件夹 {lmdb_folder} 不存在")

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if not self.use_lmdb or self.lmdb_env is None:
            raise RuntimeError("未找到有效的LMDB文件，请先运行预处理脚本生成LMDB。")
        pdb_file = str(self.pdb_files[idx])
        with self.lmdb_env.begin() as txn:
            tensor_bytes = txn.get(pdb_file.encode())
            if tensor_bytes is None:
                raise KeyError(f"{pdb_file} 不在LMDB中")
            patches = torch.load(pickle.loads(pickle.dumps(tensor_bytes)), map_location="cpu")
        return patches
