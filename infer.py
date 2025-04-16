"""
Inference script for VQTokenizer: Given a protein PDB structure file, outputs the token id tensor and the decoded reconstructed structure feature tensor.
"""

import argparse
from typing import Any

import torch
from Bio.PDB import PDBParser
from omegaconf import OmegaConf

from vqtokenizer.datasets.datapipe import angle_to_vector, calc_phi_psi, make_patches
from vqtokenizer.nn.backbone import VQTokenizer


def extract_patches_from_pdb(pdb_path: str, patch_size: int) -> torch.Tensor:
    """
    Extracts patches from a protein PDB file for model inference.
    Args:
        pdb_path (str): Path to the PDB file.
        patch_size (int): Patch size for feature extraction.
    Returns:
        torch.Tensor: Patch tensor for model input.
    Raises:
        ValueError: If not enough residues are found for patch extraction.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]
    residues = [res for res in model.get_residues() if res.has_id("N") and res.has_id("CA") and res.has_id("C")]
    if len(residues) < patch_size + 2:
        raise ValueError("Not enough residues for patch extraction.")
    phi_psi = calc_phi_psi(residues)
    angle_vecs = angle_to_vector(phi_psi)
    patches = make_patches(angle_vecs, patch_size)
    return patches


def main() -> None:
    """
    Main function for VQTokenizer inference. Loads model and config, processes input PDB, and outputs token ids and reconstructed features.
    """
    parser = argparse.ArgumentParser(description="VQTokenizer inference for protein PDB file")
    parser.add_argument("--config", type=str, default="config/train.yml", help="Path to config YAML")
    parser.add_argument("--ckpt", type=str, default="weight/final_model.ckpt", help="Path to model checkpoint")
    parser.add_argument("--pdb", type=str, required=True, help="Path to input PDB file")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    input_dim = cfg.data.patch_size * 4

    model = VQTokenizer(
        input_dim=input_dim,
        hidden_dim=cfg.model.hidden_dim,
        latent_dim=cfg.model.latent_dim,
        num_embeddings=cfg.model.num_embeddings,
        nhead=cfg.model.nhead,
        learning_rate=cfg.model.learning_rate,
    )
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    model.eval()
    model.to(device)

    patches = extract_patches_from_pdb(args.pdb, cfg.data.patch_size)
    if patches.shape[0] == 0:
        print("No valid patches extracted from PDB.")
        return
    patches = patches.to(device)

    with torch.no_grad():
        token_ids = model.encode(patches)
        recon_features = model.decode(token_ids)

    print("Token IDs tensor:", token_ids)
    print("Reconstructed features tensor:", recon_features)


if __name__ == "__main__":
    main()
