from __future__ import annotations
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import one_hot
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict
import json


import os
import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
aa_to_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
unk_index = len(AMINO_ACIDS)



try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _to_numpy(arr):
    """Convert to NumPy for internal processing."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _from_numpy(arr_np: np.ndarray, like):
    """Convert back to same type as `like`."""
    if isinstance(like, torch.Tensor):
        return torch.from_numpy(arr_np)
    return arr_np


def _save_pair_torch(dirpath: str | None, x, y):
    """Save as torch tensors with file names based on folder name."""
    if not dirpath:
        return
    os.makedirs(dirpath, exist_ok=True)
    folder_name = os.path.basename(os.path.normpath(dirpath))
    torch.save(torch.as_tensor(x), os.path.join(dirpath, f"{folder_name}_x.pt"))
    torch.save(torch.as_tensor(y), os.path.join(dirpath, f"{folder_name}_y.pt"))


def create_single_tensors(
    x_cat,
    y_true,
    single_dir: str | None = None,
    multi_dir: str | None = None,
):
    """
    Split rows into:
    - single: exactly one non -1 value in y_true
    - multi: all others
    """
    X = _to_numpy(x_cat)
    Y = _to_numpy(y_true)

    if Y.ndim == 1:
        Y = Y[:, None]

    non_neg1 = (Y != -1)
    counts = non_neg1.sum(axis=1)

    single_mask = counts == 1
    multi_mask  = ~single_mask

    x_single_np = X[single_mask]
    y_single_np = Y[single_mask]
    x_multi_np  = X[multi_mask]
    y_multi_np  = Y[multi_mask]

    # Save as torch tensors
    _save_pair_torch(single_dir, x_single_np, y_single_np)
    _save_pair_torch(multi_dir,  x_multi_np,  y_multi_np)

    # Return same type as input
    x_single = _from_numpy(x_single_np, x_cat)
    y_single = _from_numpy(y_single_np, y_true)
    x_multi  = _from_numpy(x_multi_np,  x_cat)
    y_multi  = _from_numpy(y_multi_np,  y_true)
    return x_single, y_single, x_multi, y_multi


def create_varied_tensors(
    x_multi,
    y_multi,
    monotone_dir: str | None = None,
    varied_dir: str | None = None,
):
    """
    Split multi into:
    - monotone: all non -1 labels in row are identical (0 or 1)
    - varied: others
    """
    X = _to_numpy(x_multi)
    Y = _to_numpy(y_multi)

    if Y.ndim == 1:
        Y = Y[:, None]

    non_neg1 = (Y != -1)
    valid_counts = non_neg1.sum(axis=1)

    Y_float = Y.astype(float)
    Y_float[~non_neg1] = np.nan

    row_min = np.nanmin(Y_float, axis=1)
    row_max = np.nanmax(Y_float, axis=1)

    all_same = (row_min == row_max)
    has_any = valid_counts > 0
    common_is_binary = np.isin(row_min, [0.0, 1.0])

    monotone_mask = has_any & all_same & common_is_binary
    varied_mask   = ~monotone_mask

    x_monotone_np = X[monotone_mask]
    y_monotone_np = Y[monotone_mask]
    x_varied_np   = X[varied_mask]
    y_varied_np   = Y[varied_mask]

    # Save as torch tensors
    _save_pair_torch(monotone_dir, x_monotone_np, y_monotone_np)
    _save_pair_torch(varied_dir,   x_varied_np,   y_varied_np)

    # Return same type as input
    x_monotone = _from_numpy(x_monotone_np, x_multi)
    y_monotone = _from_numpy(y_monotone_np, y_multi)
    x_varied   = _from_numpy(x_varied_np,   x_multi)
    y_varied   = _from_numpy(y_varied_np,   y_multi)
    return x_monotone, y_monotone, x_varied, y_varied

def pad_sequences(seqs, max_len=None, pad_char='X'):
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    return [seq.ljust(max_len, pad_char) for seq in seqs]

def encode_categorical(seqs: list, aa_to_index: dict) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    padded = pad_sequences(seqs, max_len=max_len, pad_char='X')
    encoded = [[aa_to_index.get(aa, unk_index) for aa in seq] for seq in padded]
    return torch.tensor(encoded, dtype=torch.long), max_len

def encode_onehot(categorical_tensor: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return one_hot(categorical_tensor, num_classes=vocab_size).float()

def build_y_true_from_species_labels( 
    df: pd.DataFrame,
    species_cols: list,
    save_tensor_path: str = 'y_true.pt',
    save_index_json_path: str = 'species_to_index.json'
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Builds y_true tensor (n_seq x no_species) and species_to_index mapping.
    Unknown labels are filled with -1. Saves species index as JSON.
    """
    assert all(col in df.columns for col in species_cols), "Some species columns are missing from DataFrame."

    # Keep row order, extract labels
    label_df = df[species_cols].copy()
    
    y_np = label_df.to_numpy()
    y_true = torch.tensor(y_np, dtype=torch.int8)  # Efficient storage

    # Save label tensor
    torch.save(y_true, save_tensor_path)

    # Save species to index mapping as JSON
    species_to_index = {species: idx for idx, species in enumerate(species_cols)}
    with open(save_index_json_path, 'w') as f:
        json.dump(species_to_index, f, indent=2)
    
    return y_true, species_to_index

if __name__ == "__main__":
    data = pd.read_csv('mls_data_full.csv')
    print(data.columns)
    species_cols = [col for col in data.columns if col in [
    'mouse', 'ihbat', 'cattle', 'human', 'willow', 'tit', 'ostrich', 'dog',
       'cat', 'monkey', 'horse', 'owl', 'boar', 'mink', 'possum', 'pangolin',
       'chicken', 'human_new', 'civet', 'rat'
        ]]
    print(species_cols)
   # y_true, species_to_index = build_y_true_from_species_labels(data, species_cols)
    x_cat= torch.load('x_categorical.pt')
    y_true = torch.load('y_true.pt')

    print(x_cat.shape)
    print(y_true.shape)

    x_single, y_single, x_multi, y_multi = create_single_tensors(x_cat, y_true, './single', './multi')

    x_monotone, y_monotone, x_varied, y_varied = create_varied_tensors(x_multi, y_multi, './monotone' , './varied')

    print(x_single.shape , y_single.shape)
    print(x_multi.shape , y_multi.shape)
    print(x_monotone.shape , y_monotone.shape)
    print(x_varied.shape , y_varied.shape)
