import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
import os
from models import SimpleBindingTransformer
import argparse
from typing import Tuple
import utils

parser = argparse.ArgumentParser(description="Load model configuration from JSON file.")
parser.add_argument("--config", type=str, required=False, default= "experiment_runs/config_0/config_0.json", help="Path to the JSON config file")
args = parser.parse_args()


with open(args.config, "r") as f:
    config = json.load(f)

print(config)

def _load_pair(dirpath: str, map_location="cpu"):
    """
    Load <folder>_x.pt and <folder>_y.pt from dirpath.
    Returns (x, y) as torch.Tensors.
    """
    folder = os.path.basename(os.path.normpath(dirpath))
    x_path = os.path.join(dirpath, f"{folder}_x.pt")
    y_path = os.path.join(dirpath, f"{folder}_y.pt")

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing file: {y_path}")

    x = torch.load(x_path, map_location=map_location)
    y = torch.load(y_path, map_location=map_location)

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)

    # Basic sanity checks
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch in {dirpath}: x has {x.shape[0]} rows, y has {y.shape[0]} rows")

    return x, y

def split_and_save(
    x: torch.Tensor,
    y: torch.Tensor,
    out_dir: str,
    config: dict,
    group_name: str = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split (x, y) into train/val/test using config['random_seed'], save to out_dir.

    Args:
        x, y: torch.Tensors with matching first dimension.
        out_dir: folder to save files.
        config: dict containing 'random_seed'.
        group_name: name for filenames (defaults to folder name).
        train_ratio: fraction for train set.
        val_ratio: fraction for val set.

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test
    """
    assert x.shape[0] == y.shape[0], "x and y must have same number of rows"
    n = x.shape[0]

    os.makedirs(out_dir, exist_ok=True)
    if group_name is None:
        group_name = os.path.basename(os.path.normpath(out_dir))

    # Calculate sizes
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0

    # Shuffle with config's seed
    g = torch.Generator()
    g.manual_seed(config["random_seed"])
    perm = torch.randperm(n, generator=g)

    x_shuf = x[perm]
    y_shuf = y[perm]

    # Split
    x_train, y_train = x_shuf[:n_train], y_shuf[:n_train]
    x_val,   y_val   = x_shuf[n_train:n_train+n_val], y_shuf[n_train:n_train+n_val]
    x_test,  y_test  = x_shuf[n_train+n_val:], y_shuf[n_train+n_val:]

    # Save with group name in filenames
    def _save_pair(split_name: str, x_t: torch.Tensor, y_t: torch.Tensor):
        torch.save(x_t, os.path.join(out_dir, f"x_{group_name}_{split_name}.pt"))
        torch.save(y_t, os.path.join(out_dir, f"y_{group_name}_{split_name}.pt"))

    _save_pair("train", x_train, y_train)
    _save_pair("val",   x_val,   y_val)
    _save_pair("test",  x_test,  y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test

single_dir   = "./single"
multi_dir    = "./multi"
monotone_dir = "./monotone"
varied_dir   = "./varied"
_map_location = "cpu"

x_single,   y_single   = _load_pair(single_dir,   map_location=_map_location)
x_multi,    y_multi    = _load_pair(multi_dir,    map_location=_map_location)
x_monotone, y_monotone = _load_pair(monotone_dir, map_location=_map_location)
x_varied,   y_varied   = _load_pair(varied_dir,   map_location=_map_location)


print(f"x_single shape: {x_single.shape}, y_single shape: {y_single.shape}")
print(f"x_multi shape: {x_multi.shape}, y_multi shape: {y_multi.shape}")
print(f"x_monotone shape: {x_monotone.shape}, y_monotone shape: {y_monotone.shape}")
print(f"x_varied shape: {x_varied.shape}, y_varied shape: {y_varied.shape}")

# Assuming config is loaded from JSON
x_single_train, y_single_train, x_single_val, y_single_val, x_single_test, y_single_test = \
    split_and_save(x_single, y_single, "single", config, group_name="single")

x_multi_train, y_multi_train, x_multi_val, y_multi_val, x_multi_test, y_multi_test = \
    split_and_save(x_multi, y_multi, "multi", config, group_name="multi")

x_monotone_train, y_monotone_train, x_monotone_val, y_monotone_val, x_monotone_test, y_monotone_test = \
    split_and_save(x_monotone, y_monotone, "monotone", config, group_name="monotone")

x_varied_train, y_varied_train, x_varied_val, y_varied_val, x_varied_test, y_varied_test = \
    split_and_save(x_varied, y_varied, "varied", config, group_name="varied")

utils.plot_label_distribution_per_column( y_single_train, "multi_train")
utils.plot_label_distribution_per_column( y_single_val, "single_val")
utils.plot_label_distribution_per_column( y_multi_train, "single_train")
utils.plot_label_distribution_per_column( y_multi_val, "multi_val")
utils.plot_label_distribution_per_column( y_monotone_train, "monotone_train")
utils.plot_label_distribution_per_column( y_monotone_val, "monotone_val")
utils.plot_label_distribution_per_column( y_varied_train, "varied_train")
utils.plot_label_distribution_per_column( y_varied_val, "varied_val")


x_train = torch.cat([ x_varied_train, x_monotone_train], dim=0)
y_train = torch.cat([ y_varied_train, y_monotone_train], dim=0)
print(f"Combined x_train shape (single+varied): {tuple(x_train.shape)}")
utils.plot_label_distribution_per_column(y_train, "combined_train")  
print(x_train.shape, y_train.shape  )


def print_first_five(name, x, y):
    print(f"\n=== {name.upper()} ===")
    n = min(5, x.shape[0])
    for i in range(n):
        print(f"Row {i} | Y = {y[i].tolist()}")



with open('species_to_index.json', 'r') as f:
    species_to_index = json.load(f)
NUM_SPECIES = len(species_to_index)
print(species_to_index)
model = None
if config["model_architecture"] == "transformer":
    model = SimpleBindingTransformer(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        transformer_dim=config["transformer_dim"],
        num_layers=config["num_layers"],
        num_species=NUM_SPECIES
    ).to(config["device"])


optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
criterion = nn.BCEWithLogitsLoss(reduction='none')

