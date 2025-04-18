"""
Preprocessing script: Precompute patches for all PDB files and save them to LMDB, and save the MD5 of the LMDB.
Usage: python tools/precompute_patches_lmdb.py --config config/train.yml --lmdb_folder ./data/lmdb
"""

import argparse
import concurrent.futures
import hashlib
from pathlib import Path

import torch
from Bio.PDB import PDBParser
from Bio.PDB.Residue import Residue
from Bio.PDB.vectors import calc_dihedral
from omegaconf import OmegaConf
from tqdm import tqdm

import lmdb


# --- Functions related to patch computation ---
def calc_phi_psi(residues: list[Residue]) -> torch.Tensor:
    phis = []
    psis = []
    for i in range(1, len(residues) - 1):
        try:
            n1 = residues[i - 1]["C"].get_vector()
            c = residues[i]["C"].get_vector()
            ca = residues[i]["CA"].get_vector()
            n = residues[i]["N"].get_vector()
            n2 = residues[i + 1]["N"].get_vector()
            phi = calc_dihedral(n1, n, ca, c)
            psi = calc_dihedral(n, ca, c, n2)
            phis.append(phi)
            psis.append(psi)
        except Exception:
            continue
    return torch.stack([torch.tensor(phis), torch.tensor(psis)], axis=1)  # shape: (N, 2)


def angle_to_vector(angles: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [torch.cos(angles[:, 0]), torch.sin(angles[:, 0]), torch.cos(angles[:, 1]), torch.sin(angles[:, 1])], axis=-1
    )


def make_patches(vecs: torch.Tensor, patch_size: int = 3) -> torch.Tensor:
    num_patches = len(vecs) - patch_size + 1
    if num_patches <= 0:
        return torch.zeros((0, patch_size * vecs.shape[1]), dtype=torch.float32)
    patches = torch.zeros((num_patches, patch_size * vecs.shape[1]), dtype=torch.float32)
    for i in range(num_patches):
        patches[i] = vecs[i : i + patch_size].reshape(-1)
    return patches.float()


def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def valid_pdb_residues(pdb_file, patch_size):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]
    residues = [res for res in model.get_residues() if res.has_id("N") and res.has_id("CA") and res.has_id("C")]
    return residues if len(residues) >= patch_size + 2 else None


def process_pdb_file(args):
    pdb_file, patch_size = args
    residues = valid_pdb_residues(pdb_file, patch_size)
    if residues is None:
        return None
    phi_psi = calc_phi_psi(residues)
    angle_vecs = angle_to_vector(phi_psi)
    patches = make_patches(angle_vecs, patch_size)
    tensor_bytes = torch.save(patches, _return_bytes=True)
    return (str(pdb_file), tensor_bytes)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute patches for all PDB files and save them to LMDB, and save the MD5 of the LMDB."
    )
    parser.add_argument("--config", type=str, default="../config/train.yml", help="Path to config file")
    parser.add_argument("--lmdb_folder", type=str, required=True, help="Output LMDB folder path")
    args = parser.parse_args()
    # Read config
    config = OmegaConf.load(args.config)
    data_cfg = config["data"]
    pdb_dir = data_cfg["pdb_dir"]
    patch_size = data_cfg["patch_size"]

    lmdb_folder = Path(args.lmdb_folder)
    lmdb_folder.mkdir(parents=True, exist_ok=True)
    lmdb_path = lmdb_folder / "train.lmdb"

    num_workers = data_cfg.get("num_workers", 1)
    pdb_files = list(Path(pdb_dir).glob("*.pdb"))
    tasks = [(pdb_file, patch_size) for pdb_file in pdb_files]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(process_pdb_file, tasks), total=len(tasks), desc="Processing PDBs"):
            if result is not None:
                results.append(result)

    env = lmdb.open(str(lmdb_path), map_size=1024**4)
    with env.begin(write=True) as txn:
        for pdb_file, tensor_bytes in tqdm(results, desc="Writing to LMDB"):
            txn.put(pdb_file.encode(), tensor_bytes)
    env.close()

    # Compute MD5 for the actual LMDB data file (data.mdb)
    data_mdb_path = lmdb_path / "data.mdb"
    md5 = compute_md5(data_mdb_path)
    md5_path = data_mdb_path.with_suffix(".md5")
    with open(md5_path, "w") as f:
        f.write(md5)
    print(f"LMDB saved: {lmdb_path}\nMD5: {md5}\nMD5 file: {md5_path}")


if __name__ == "__main__":
    main()
