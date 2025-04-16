"""
vqtokenizer/datasets/datapipe.py
Description:
This module implements loading of protein structure datasets, angle calculation, vector conversion, and data modules.
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from Bio.PDB import PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.PDB.vectors import Vector, calc_dihedral
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


# Calculate phi and psi dihedral angles for a list of residues
def calc_phi_psi(residues: list[Residue]) -> torch.Tensor:
    """
    Calculate phi and psi dihedral angles for a list of residues.
    Args:
        residues (List[Residue]): List of Biopython Residue objects.
    Returns:
        torch.Tensor: Tensor of shape (N, 2), where N is the number of valid residues, columns are [phi, psi].
    """
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


# Convert angles to vector representation
def angle_to_vector(angles: torch.Tensor) -> torch.Tensor:
    """
    Convert angles to vector representation (cos, sin for each angle).
    Args:
        angles (torch.Tensor): [N, 2] tensor of angles.
    Returns:
        torch.Tensor: [N, 4] tensor, each row is [cos(phi), sin(phi), cos(psi), sin(psi)].
    """
    return torch.stack(
        [torch.cos(angles[:, 0]), torch.sin(angles[:, 0]), torch.cos(angles[:, 1]), torch.sin(angles[:, 1])], axis=-1
    )  # (N, 4)


# Create patches from vectors
def make_patches(vecs: torch.Tensor, patch_size: int = 3) -> torch.Tensor:
    """
    Create sliding window patches from input vectors.
    Args:
        vecs (torch.Tensor): [N, D] input vectors.
        patch_size (int): Number of consecutive vectors per patch.
    Returns:
        torch.Tensor: [num_patches, patch_size * D] patches.
    """
    num_patches = len(vecs) - patch_size + 1
    if num_patches <= 0:
        return torch.zeros((0, patch_size * vecs.shape[1]), dtype=torch.float32)

    patches = torch.zeros((num_patches, patch_size * vecs.shape[1]), dtype=torch.float32)
    for i in range(num_patches):
        patches[i] = vecs[i : i + patch_size].reshape(-1)

    return patches.float()


def _check_pdb_file(pdb_file, patch_size):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("check", pdb_file)
        model = structure[0]
        residues = [res for res in model.get_residues() if res.has_id("N") and res.has_id("CA") and res.has_id("C")]
        if len(residues) >= patch_size + 2:
            return str(pdb_file)
    except Exception:
        return None


class ProteinStructureDataset(Dataset):
    """
    Dataset for protein structures, extracts patches of angle vectors from PDB files.
    Args:
        pdb_dir (str or Path): Directory containing PDB files.
        patch_size (int): Number of residues per patch.
        num_workers (int): Number of worker processes for parallel processing.
    Returns:
        Each item is a tensor of shape (num_patches, patch_size * 4).
    """

    def __init__(self, pdb_dir: str, patch_size: int = 3, num_workers: int = 4) -> None:
        self.pdb_files: list[Path] = list(Path(pdb_dir).glob("*.pdb"))
        self.patch_size: int = patch_size
        self.parser: PDBParser = PDBParser(QUIET=True)
        self.num_workers = num_workers

        cache_file = Path(pdb_dir) / "valid_pdb_files.txt"
        if cache_file.exists():
            with open(cache_file) as f:
                self.pdb_files = [Path(line.strip()) for line in f if line.strip()]
            print(f"Loaded {len(self.pdb_files)} valid PDB files from cache")
            return

        # 并行检查
        valid_files: list[Path] = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(_check_pdb_file, pdb_file, patch_size) for pdb_file in self.pdb_files]
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    valid_files.append(Path(result))

        self.pdb_files = valid_files
        print(f"Loaded {len(self.pdb_files)} valid PDB files")
        # 缓存结果
        with open(cache_file, "w") as f:
            for pdb_file in self.pdb_files:
                f.write(str(pdb_file) + "\n")

    def __len__(self) -> int:
        """
        Returns the number of valid PDB files.
        """
        return len(self.pdb_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Loads a PDB file, extracts residue angle vectors, and returns patches.
        Args:
            idx (int): Index of the PDB file.
        Returns:
            torch.Tensor: [num_patches, patch_size * 4] patches for the protein.
        """
        pdb_file = self.pdb_files[idx]

        # Parse PDB file
        structure: Structure = self.parser.get_structure("protein", pdb_file)
        model: Model = structure[0]
        residues: list[Residue] = [
            res for res in model.get_residues() if res.has_id("N") and res.has_id("CA") and res.has_id("C")
        ]

        # Calculate dihedral angles
        phi_psi: torch.Tensor = calc_phi_psi(residues)

        # Convert to vector representation
        angle_vecs: torch.Tensor = angle_to_vector(phi_psi)

        # Create patches
        patches: torch.Tensor = make_patches(angle_vecs, self.patch_size)

        return patches  # shape: (num_patches, patch_size * 4)


# Custom collate function to merge all patches in a batch
def collate_protein_patches(batch: list[torch.Tensor]) -> torch.Tensor:
    """
    Collate function to concatenate all patches from a batch of proteins.
    Args:
        batch (List[torch.Tensor]): List of [num_patches, patch_size * 4] tensors.
    Returns:
        torch.Tensor: [total_patches, patch_size * 4] concatenated patches.
    """
    all_patches = torch.cat(batch, dim=0)
    return all_patches


class ProteinDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for protein structure patch extraction.
    Args:
        pdb_dir (str or Path): Directory containing PDB files.
        patch_size (int): Number of residues per patch.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker processes for DataLoader.
    """

    def __init__(self, pdb_dir: str, patch_size: int = 3, batch_size: int = 64, num_workers: int = 4) -> None:
        super().__init__()
        self.pdb_dir = pdb_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset: ProteinStructureDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """
        Setup the dataset and split into train/validation sets.
        """
        # Create dataset
        self.dataset = ProteinStructureDataset(self.pdb_dir, self.patch_size, self.num_workers)

        # Split dataset into training and validation sets
        dataset_size = len(self.dataset)
        train_size = int(dataset_size * 0.8)
        val_size = dataset_size - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=1,  # Each batch loads all patches from one protein
            shuffle=True,
            collate_fn=collate_protein_patches,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation DataLoader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=1,  # Each batch loads all patches from one protein
            shuffle=False,
            collate_fn=collate_protein_patches,
            num_workers=self.num_workers,
        )
