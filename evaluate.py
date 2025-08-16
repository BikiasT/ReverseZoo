import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
import json
from models import SimpleBindingTransformer
import argparse
from torchmetrics.classification import MatthewsCorrCoef
import os
import utils
parser = argparse.ArgumentParser(description="Load model configuration from JSON file.")
parser.add_argument("--config", type=str, required=False, default= "experiment_runs/config_0/config_0.json", help="Path to the JSON config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)
# Load species mapping
with open('species_to_index.json') as f:
    species_to_index = json.load(f)
SPECIES = list(species_to_index.keys())
NUM_SPECIES = len(SPECIES)

if config["model_architecture"] == "transformer":
    model = SimpleBindingTransformer(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        transformer_dim=config["transformer_dim"],
        num_layers=config["num_layers"],
        num_species=NUM_SPECIES
    ).to(config["device"])
#model.load_state_dict(torch.load('model.pt'))
model.eval()

def load_all_test_sets(base_dirs, map_location="cpu"):
    results = []
    for group, folder in base_dirs.items():
        x_path = os.path.join(folder, f"x_{group}_test.pt")
        y_path = os.path.join(folder, f"y_{group}_test.pt")
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError(f"Missing test files for group '{group}' in {folder}")
        x_test = torch.load(x_path, map_location=map_location)
        y_test = torch.load(y_path, map_location=map_location)
        results.extend([x_test, y_test])
    return tuple(results)

if __name__ == "__main__":

    base_dirs = {
        "single": "./single",
        "multi": "./multi",
        "monotone": "./monotone",
        "varied": "./varied"
    }

    (x_single_test, y_single_test,
     x_multi_test, y_multi_test,
     x_monotone_test, y_monotone_test,
     x_varied_test, y_varied_test) = load_all_test_sets(base_dirs, map_location="cpu")

    # Optional: quick sanity check
    print(f"x_single_test shape: {x_single_test.shape}, y_single_test shape: {y_single_test.shape}")
    print(f"x_multi_test shape: {x_multi_test.shape}, y_multi_test shape: {y_multi_test.shape}")
    print(f"x_monotone_test shape: {x_monotone_test.shape}, y_monotone_test shape: {y_monotone_test.shape}")
    print(f"x_varied_test shape: {x_varied_test.shape}, y_varied_test shape: {y_varied_test.shape}")

    utils.plot_label_distribution_per_column(y_single_test, "single_test")
    utils.plot_label_distribution_per_column(y_monotone_test, "monotone_test")  
    utils.plot_label_distribution_per_column(y_varied_test, "varied_test")
    utils.plot_label_distribution_per_column(y_multi_test, "multi_test")
