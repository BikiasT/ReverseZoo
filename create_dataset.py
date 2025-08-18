import json
import torch
import numpy as np
import matplotlib.pyplot as plt

with open('species_to_index.json', 'r') as f:
    species_dict = json.load(f)  # data is now a Python dict


# ========== Configuration ==========
EXCLUDE_SPECIES = []  # ← e.g. skip these species
OUTPUT_FIGURE = "binding_distribution.png"
# ===================================

# Load data
y_varied = torch.load('varied/varied_y.pt')
x_varied = torch.load('varied/varied_x.pt')

y_multi = torch.load('multi/multi_y.pt')
x_multi = torch.load('multi/multi_x.pt')

y_single = torch.load('single/single_y.pt')
x_single = torch.load('single/single_x.pt')    

print(f"Loaded tensors: y_varied {y_varied.shape}, x_varied {x_varied.shape}")
print(f"Loaded tensors: y_multi {y_multi.shape}, x_multi {x_multi.shape}")
print(f"Loaded tensors: y_single {y_single.shape}, x_single {x_single.shape}")

# Load species index dictionary
assert 'species_dict' in globals(), "Please define `species_dict` before running."

num_species = len(species_dict)
balanced_datasets = {}

# Tracking for visualization
label_hist = {}
from collections import defaultdict

# Add tracking
dataset_hist = defaultdict(lambda: {'varied': 0, 'multi': 0, 'single': 0})

# Start balancing loop
for species, index in species_dict.items():
    if species in EXCLUDE_SPECIES:
        print(f"🚫 Skipping species: {species}")
        continue

    print(f"\n🔍 Balancing species: {species} (index {index})")

    pos_indices, neg_indices = [], []

    # Varied
    pos_var = np.where(y_varied[:, index] == 1)[0]
    neg_var = np.where(y_varied[:, index] == 0)[0]
    pos_indices += [('varied', 0, i) for i in pos_var]
    neg_indices += [('varied', 0, i) for i in neg_var]

    # Multi
    if len(pos_indices) < 20000:
        pos_multi = np.where(y_multi[:, index] == 1)[0]
        pos_indices += [('multi', 1, i) for i in pos_multi]
    if len(neg_indices) < 20000:
        neg_multi = np.where(y_multi[:, index] == 0)[0]
        neg_indices += [('multi', 1, i) for i in neg_multi]

    # Single
    if len(pos_indices) < 20000:
        pos_single = np.where(y_single[:, index] == 1)[0]
        pos_indices += [('single', 2, i) for i in pos_single]
    if len(neg_indices) < 20000:
        neg_single = np.where(y_single[:, index] == 0)[0]
        neg_indices += [('single', 2, i) for i in neg_single]

    pos_indices = pos_indices[:20000]
    neg_indices = neg_indices[:20000]
    all_indices = pos_indices + neg_indices

    x_balanced = []
    y_balanced = []
    source_vector = []
    source_dataset = []

    for dataset_name, dataset_idx, sample_idx in all_indices:
        if dataset_idx == 0:
            x = x_varied[sample_idx]
            y = y_varied[sample_idx, index]
        elif dataset_idx == 1:
            x = x_multi[sample_idx]
            y = y_multi[sample_idx, index]
        else:
            x = x_single[sample_idx]
            y = y_single[sample_idx, index]

        x_balanced.append(x)
        y_balanced.append(y)
        source_vector.append(index)
        source_dataset.append(dataset_name)
        dataset_hist[species][dataset_name] += 1

    x_balanced = torch.stack(x_balanced)
    y_balanced = torch.tensor(y_balanced).unsqueeze(1)
    source_vector = torch.tensor(source_vector)

    # Shuffle while maintaining alignment
    indices = torch.randperm(x_balanced.size(0))
    x_balanced = x_balanced[indices]
    y_balanced = y_balanced[indices]
    source_vector = source_vector[indices]
    source_dataset = np.array(source_dataset)[indices.numpy()]
    # After creating y_balanced
    num_ones = int((y_balanced == 1).sum().item())
    num_zeros = int((y_balanced == 0).sum().item())

    label_hist[species] = {
        '1': num_ones,
        '0': num_zeros
    }
    # Save
    balanced_datasets[species] = {
        'x': x_balanced,
        'y': y_balanced,
        'source_species_idx': source_vector,
        'source_dataset_name': source_dataset
    }

    print(f"✅ Balanced {species}: 1s={(y_balanced==1).sum().item()} | 0s={(y_balanced==0).sum().item()}")
    print(f"x shape: {x_balanced.shape}, y shape: {y_balanced.shape}")

# ========== Plot Class Distribution ==========
species_list = list(label_hist.keys())
ones = [label_hist[s]['1'] for s in species_list]
zeros = [label_hist[s]['0'] for s in species_list]

x = np.arange(len(species_list))
bar_width = 0.4

plt.figure(figsize=(12, 6))
plt.bar(x - bar_width/2, zeros, width=bar_width, label='Class 0 (no bind)', color='skyblue')
plt.bar(x + bar_width/2, ones, width=bar_width, label='Class 1 (bind)', color='salmon')

plt.xticks(x, species_list, rotation=45, ha='right')
plt.xlabel("Species")
plt.ylabel("Sample Count")
plt.title("Binding Label Distribution per Species (after balancing)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, dpi=300)
print(f"\n📊 Binding distribution plot saved to {OUTPUT_FIGURE}")
plt.show()

# ========== Plot Dataset Source Distribution ==========
species_list = list(dataset_hist.keys())
dataset_counts = {
    "varied": [dataset_hist[s]["varied"] for s in species_list],
    "multi": [dataset_hist[s]["multi"] for s in species_list],
    "single": [dataset_hist[s]["single"] for s in species_list]
}

x = np.arange(len(species_list))
bar_width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - bar_width, dataset_counts["varied"], width=bar_width, label="varied", color='mediumseagreen')
plt.bar(x, dataset_counts["multi"], width=bar_width, label="multi", color='steelblue')
plt.bar(x + bar_width, dataset_counts["single"], width=bar_width, label="single", color='orchid')

plt.xticks(x, species_list, rotation=45, ha='right')
plt.xlabel("Species")
plt.ylabel("Sample Count")
plt.title("Dataset Origin per Species (used in balancing)")
plt.legend()
plt.tight_layout()
plt.savefig("dataset_source_distribution.png", dpi=300)
print(f"📊 Dataset source distribution saved to dataset_source_distribution.png")

from pathlib import Path

print("\n💾 Saving final stacked dataset to ./dataset/")
dataset_dir = Path("dataset")
dataset_dir.mkdir(parents=True, exist_ok=True)

# Stack all species
X_all = torch.cat([v["x"] for v in balanced_datasets.values()], dim=0)
Y_all = torch.cat([v["y"] for v in balanced_datasets.values()], dim=0)
species_idx = torch.cat([v["source_species_idx"] for v in balanced_datasets.values()], dim=0)
species_source = np.concatenate([v["source_dataset_name"] for v in balanced_datasets.values()], axis=0)

# Save tensors
torch.save(X_all, dataset_dir / "X_all.pt")
torch.save(Y_all, dataset_dir / "Y_all.pt")
torch.save(species_idx, dataset_dir / "species_idx.pt")
np.save(dataset_dir / "species_source.npy", species_source)

# Save species column names (for decoding later)
with open(dataset_dir / "species_columns.json", "w") as f:
    json.dump(list(species_dict.keys()), f, indent=2)

print(f"✅ Saved: X_all.pt [shape {X_all.shape}]")
print(f"✅ Saved: Y_all.pt [shape {Y_all.shape}]")
print(f"✅ Saved: species_idx.pt [shape {species_idx.shape}]")
print(f"✅ Saved: species_source.npy [shape {species_source.shape}]")
print(f"✅ Saved: species_columns.json")