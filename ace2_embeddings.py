#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List
import numpy as np
import torch
import matplotlib.pyplot as plt

import umap  # pip install umap-learn

# ESM Cambrian (open models)
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def load_species_json(path: str) -> Dict[str, dict]:
    with open(path, "r") as f:
        return json.load(f)


@torch.no_grad()
def embed_with_esmc(sequence: str, client: ESMC, device: str = "cuda") -> np.ndarray:
    """
    Returns a single 1D embedding vector for the input sequence using ESM Cambrian.
    We request per-position embeddings and mean-pool over the sequence length.
    """
    protein = ESMProtein(sequence=sequence)
    toks = client.encode(protein)  # tokenized tensor (batch of 1)
    toks = toks.to(device) if hasattr(toks, "to") else toks

    # Ask model for sequence logits + return_embeddings
    out = client.logits(toks, LogitsConfig(sequence=True, return_embeddings=True))

    # `out.embeddings` can be a tensor or a wrapped object depending on version.
    # Normalize to a torch.Tensor and mean-pool across positions.
    emb = out.embeddings
    if isinstance(emb, (list, tuple)):
        emb = emb[0]
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)

    # Expected shape: (B, L, D) or (L, D). Handle both.
    if emb.dim() == 3:
        emb = emb.mean(dim=1)   # (B, D)
        emb = emb[0]            # (D,)
    elif emb.dim() == 2:
        emb = emb.mean(dim=0)   # (D,)
    # else: (D,) already

    return emb.detach().cpu().numpy().astype(np.float32)


def umap_plot(emb: np.ndarray, labels: List[str], out_path: str, seed: int = 42, title: str = "UMAP of ACE2 embeddings"):
    """
    emb: (N, D), labels: list[str] length N
    Saves 2D scatter with species names annotated above the points.
    """
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=10, min_dist=0.1, metric="euclidean")
    pts = reducer.fit_transform(emb)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(pts[:, 0], pts[:, 1])
    for i, lbl in enumerate(labels):
        plt.annotate(lbl, (pts[i, 0], pts[i, 1]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved UMAP plot to: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Embed ACE2 with ESM Cambrian (local), update JSON, and plot UMAP.")
    ap.add_argument("--json", required=False, default = "updated_species_dict.json", help="Path to species_dict JSON.")
    ap.add_argument("--out_json", default="./sp_dct.json", help="Where to save updated JSON (default: overwrite input).")
    ap.add_argument("--model", default="esmc_6b", help="ESM Cambrian checkpoint (e.g., esmc_300m, esmc_600m, esmc_6b).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    ap.add_argument("--outdir", default="plots", help="Folder to save the UMAP plot.")
    ap.add_argument("--outfile", default="ace2_umap.png", help="UMAP image filename.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for UMAP.")
    args = ap.parse_args()

    species_dict = load_species_json(args.json)
    species_names = list(species_dict.keys())

    # Load ESM Cambrian model locally (weights pulled from Hugging Face on first run)
    client = ESMC.from_pretrained(args.model).to(args.device)  # e.g. "esmc_300m"
    client.eval()

    # Embed each sequence and attach to dict
    embeddings = []
    for sp in species_names:
        seq = species_dict[sp]["ace2_seq"]
        vec = embed_with_esmc(seq, client, device=args.device)
        species_dict[sp]["embedding"] = vec.tolist()  # JSON-serializable
        embeddings.append(vec)

    X = np.stack(embeddings, axis=0)  # (N, D)

    # Save updated JSON
    out_json = args.out_json or args.json
    with open(out_json, "w") as f:
        json.dump(species_dict, f, indent=2)
    print(f"Saved updated species_dict with embeddings to: {out_json}")

    # UMAP plot with labels above points
    out_path = os.path.join(args.outdir, args.outfile)
    umap_plot(X, species_names, out_path, seed=args.seed, title=f"UMAP of ACE2 embeddings ({args.model})")


if __name__ == "__main__":
    main()
