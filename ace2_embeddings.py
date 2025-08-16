#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt

# UMAP + clustering
import umap  # pip install umap-learn
from sklearn.cluster import KMeans, DBSCAN

# ESM Cambrian (open models; local weights pulled from HF)
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def load_species_json(path: str) -> Dict[str, dict]:
    with open(path, "r") as f:
        return json.load(f)


@torch.no_grad()
def embed_with_esmc(sequence: str, client: ESMC, device: str = "cuda") -> np.ndarray:
    """
    Produce a single 1D embedding for a sequence using ESM-C.
    We request per-position embeddings and mean-pool over length.
    """
    protein = ESMProtein(sequence=sequence)
    toks = client.encode(protein)
    toks = toks.to(device) if hasattr(toks, "to") else toks

    out = client.logits(toks, LogitsConfig(sequence=True, return_embeddings=True))
    emb = out.embeddings
    if isinstance(emb, (list, tuple)):
        emb = emb[0]
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)

    if emb.dim() == 3:           # (B, L, D)
        emb = emb.mean(dim=1)[0] # (D,)
    elif emb.dim() == 2:         # (L, D)
        emb = emb.mean(dim=0)    # (D,)
    else:                         # (D,)
        pass

    return emb.detach().cpu().numpy().astype(np.float32)


def umap_2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=10, min_dist=0.1, metric="euclidean")
    return reducer.fit_transform(X)  # (N, 2)


def save_umap_labeled(pts: np.ndarray, labels: List[str], out_path: str, title: str):
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
    print(f"Saved labeled UMAP: {out_path}")


def cluster_umap(
    pts: np.ndarray,
    method: str = "kmeans",
    k: int = 3,
    eps: float = 0.5,
    min_samples: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """
    Return cluster labels for UMAP points.
    - kmeans: integer labels [0..k-1]
    - dbscan: integer labels, with -1 = noise
    """
    if method == "kmeans":
        k = max(1, int(k))
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        return km.fit_predict(pts)
    elif method == "dbscan":
        db = DBSCAN(eps=eps, min_samples=min_samples)
        return db.fit_predict(pts)
    else:
        raise ValueError("method must be 'kmeans' or 'dbscan'")


def save_umap_clusters(
    pts: np.ndarray,
    clusters: np.ndarray,
    labels: List[str],
    out_path: str,
    title: str,
    include_legend: bool = True,
):
    """
    Save a UMAP scatter colored by cluster. Legend lists clusters.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    uniq = sorted(set(clusters))
    # choose palette
    cmap = plt.cm.get_cmap("tab20")
    colors = {c: cmap(i / max(1, len(uniq)-1)) for i, c in enumerate(uniq)}

    plt.figure(figsize=(8, 6))
    for c in uniq:
        mask = clusters == c
        label = f"cluster {c}" if c != -1 else "noise (-1)"
        plt.scatter(pts[mask, 0], pts[mask, 1], label=label, color=colors[c])
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    if include_legend:
        plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved clustered UMAP: {out_path}")


def save_cluster_csv(species: List[str], clusters: np.ndarray, out_csv: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("species,cluster\n")
        for sp, cl in zip(species, clusters):
            f.write(f"{sp},{cl}\n")
    print(f"Saved cluster assignments: {out_csv}")



def main():
    ap = argparse.ArgumentParser(description="Embed ACE2 with ESM Cambrian (local), update JSON, and plot UMAP.")
    ap.add_argument("--json", required=False, default = "updated_species_dict.json", help="Path to species_dict JSON.")
    ap.add_argument("--out_json", default="./sp_dct.json", help="Where to save updated JSON (default: overwrite input).")
    ap.add_argument("--model", default="esmc_600m", help="ESM Cambrian checkpoint (e.g., esmc_300m, esmc_600m, esmc_6b).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    ap.add_argument("--outdir", default="plots", help="Folder to save the UMAP plot.")
    ap.add_argument("--outfile", default="ace2_umap.png", help="UMAP image filename.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for UMAP.")
    ap.add_argument("--cluster_method", choices=["kmeans", "dbscan"], default="kmeans")
    ap.add_argument("--k", type=int, default=3, help="Number of clusters for kmeans.")
    ap.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps.")
    ap.add_argument("--min_samples", type=int, default=2, help="DBSCAN min_samples.")
    ap.add_argument("--save_csv", default=None, help="Optional CSV path to save species->cluster mapping.")
    args = ap.parse_args()

    species_dict = load_species_json(args.json)
    species_names = list(species_dict.keys())

    # Load ESM Cambrian locally
    client = ESMC.from_pretrained(args.model).to(args.device)
    client.eval()

    # Embed each species and attach to JSON
    embeddings = []
    for sp in species_names:
        seq = species_dict[sp]["ace2_seq"]
        vec = embed_with_esmc(seq, client, device=args.device)
        species_dict[sp]["embedding"] = vec.tolist()
        embeddings.append(vec)
    X = np.stack(embeddings, axis=0)  # (N, D)

    # Save updated JSON
    out_json = args.out_json or args.json
    with open(out_json, "w") as f:
        json.dump(species_dict, f, indent=2)
    print(f"Saved updated species_dict with embeddings to: {out_json}")

    # UMAP
    pts = umap_2d(X, seed=args.seed)

    # 1) Labeled plot
    labeled_path = os.path.join(args.outdir, f"ace2_umap_{args.model}_labeled.png")
    save_umap_labeled(pts, species_names, labeled_path, f"UMAP of ACE2 (ESM-C {args.model})")

    # 2) Clustered plot
    clusters = cluster_umap(
        pts,
        method=args.cluster_method,
        k=args.k,
        eps=args.eps,
        min_samples=args.min_samples,
        seed=args.seed,
    )
    clusters_path = os.path.join(args.outdir, f"ace2_umap_{args.model}_clusters_{args.cluster_method}.png")
    save_umap_clusters(
        pts, clusters, species_names, clusters_path,
        title=f"UMAP Clusters ({args.cluster_method}, model={args.model})",
        include_legend=True
    )

    # Optional CSV of assignments
    if args.save_csv:
        save_cluster_csv(species_names, clusters, args.save_csv)


if __name__ == "__main__":
    main()