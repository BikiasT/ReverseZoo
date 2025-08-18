import torch
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import pandas as pd

# ========== CONFIG ==========
DATASET_DIR = Path("dataset")
EXCLUDE_SPECIES = ['possum', 'rat','tit','ihbat','willow']  # ← test-only species ['possum', 'rat','tit','ihbat','willow']
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
# ============================

# Load data
X_all = torch.load(DATASET_DIR / "X_all.pt")
Y_all = torch.load(DATASET_DIR / "Y_all.pt")
species_idx = torch.load(DATASET_DIR / "species_idx.pt")
species_source = np.load(DATASET_DIR / "species_source.npy", allow_pickle=True)

# Load species columns
with open(DATASET_DIR / "species_columns.json", "r") as f:
    species_list = json.load(f)

species_to_index = {s: i for i, s in enumerate(species_list)}
exclude_species_indices = {species_to_index[s] for s in EXCLUDE_SPECIES}
print(Y_all.shape)
print(species_idx.shape)
# Split into test-only and eligible-for-split
test_mask = torch.tensor([idx.item() in exclude_species_indices for idx in species_idx])
trainval_mask = ~test_mask

X_test = X_all[test_mask]
Y_test = Y_all[test_mask]
species_test = species_idx[test_mask]

X_trainval = X_all[trainval_mask]
Y_trainval = Y_all[trainval_mask]
species_trainval = species_idx[trainval_mask]

# Train/val split
X_train, X_val, Y_train, Y_val, species_train, species_val = train_test_split(
    X_trainval, Y_trainval, species_trainval,
    test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
    random_state=SEED,
    stratify=species_trainval
)

import matplotlib.pyplot as plt

def plot_distribution(name, y_tensor, species_idx_tensor, save_path):
    species_names = np.array(species_list)
    df = pd.DataFrame({
        "species": species_names[species_idx_tensor.numpy()],
        "label": y_tensor.squeeze().numpy()
    })

    counts = df.groupby(["species", "label"]).size().unstack(fill_value=0)
    species_sorted = counts.index.tolist()
    zeros = counts.get(0, pd.Series(0, index=counts.index))
    ones = counts.get(1, pd.Series(0, index=counts.index))

    x = np.arange(len(species_sorted))
    bar_width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x - bar_width/2, zeros, width=bar_width, label="Class 0 (no bind)", color='skyblue')
    plt.bar(x + bar_width/2, ones, width=bar_width, label="Class 1 (bind)", color='salmon')

    plt.xticks(x, species_sorted, rotation=45, ha='right')
    plt.xlabel("Species")
    plt.ylabel("Sample Count")
    plt.title(f"{name} Set Class Distribution per Species")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📈 Saved {name} distribution plot to {save_path}")


# ======= Generate Plots =======
plot_distribution("Train", Y_train, species_train, DATASET_DIR / "train_distribution.png")
plot_distribution("Valid", Y_val, species_val, DATASET_DIR / "val_distribution.png")
plot_distribution("Test", Y_test, species_test, DATASET_DIR / "test_distribution.png")

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from collections import defaultdict
import random

# ====== Constants ======
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
aa_to_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
unk_index = len(AMINO_ACIDS)
vocab_size = unk_index + 1

# ====== Dataset that assumes X is already encoded ======
class PreEncodedBindingDataset(Dataset):
    def __init__(self, x_encoded: torch.Tensor, y: torch.Tensor, species_idx: torch.Tensor):
        """
        x_encoded: Tensor [N, D] - already one-hot encoded and flattened
        y: Tensor [N, 1] or [N] - labels
        species_idx: Tensor [N] - index of species
        """
        self.x = x_encoded.float()
        self.y = y.float().squeeze()
        self.species_idx = species_idx.long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.species_idx[idx]

# ====== Balanced Sampler (unchanged) ======
class BalancedSpeciesSampler(Sampler):
    def __init__(self, labels: torch.Tensor, species_indices: torch.Tensor, samples_per_bin: int = 10):
        self.labels = labels.int()
        self.species = species_indices.int()
        self.samples_per_bin = samples_per_bin
        self.bin_indices = defaultdict(list)

        for idx, (y, s) in enumerate(zip(self.labels, self.species)):
            self.bin_indices[(s.item(), y.item())].append(idx)

        self.valid_bins = list(self.bin_indices.keys())

    def __iter__(self):
        selected_indices = []
        for bin_key in self.valid_bins:
            pool = self.bin_indices[bin_key]
            if len(pool) >= self.samples_per_bin:
                selected = random.sample(pool, self.samples_per_bin)
            else:
                selected = random.choices(pool, k=self.samples_per_bin)
            selected_indices.extend(selected)
        random.shuffle(selected_indices)
        return iter(selected_indices)

    def __len__(self):
        return len(self.valid_bins) * self.samples_per_bin

# ====== Dataloader Constructor ======
def get_balanced_dataloader_from_encoded(
    x_encoded: torch.Tensor,
    y: torch.Tensor,
    species_idx: torch.Tensor,
    samples_per_bin: int = 10,
    batch_size: int = 32
):
    dataset = PreEncodedBindingDataset(x_encoded, y, species_idx)
    sampler = BalancedSpeciesSampler(y, species_idx, samples_per_bin)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


species_train = species_idx

train_loader = get_balanced_dataloader_from_encoded(
    x_encoded=X_train,
    y=Y_train,
    species_idx=species_train,
    samples_per_bin=100,
    batch_size=512
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# === Constants ===
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
vocab_size = len(AMINO_ACIDS) + 1  # +1 for unknown
num_species = len(set(species_idx.tolist()))
seq_len = X_all.shape[1]
input_dim = seq_len * vocab_size + num_species
# === One-hot function for amino acid sequences ===
def one_hot_encode_x(x_batch):
    x_batch[x_batch >= vocab_size] = vocab_size - 1  # cap to valid range
    x_onehot = F.one_hot(x_batch, num_classes=vocab_size).float()
    return x_onehot.view(x_batch.size(0), -1)  # Flatten

# === One-hot function for species ===
def one_hot_encode_species(sp_batch):
    return F.one_hot(sp_batch, num_classes=num_species).float()

# === Model ===
class LogisticModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)

# === Initialize Model ===
model = LogisticModel(input_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# === Training Loop (1 epoch example) ===
model.train()
for epoch in range(50):  # Just one epoch for demonstration
    print(f"Epoch {epoch + 1}")
    for batch_idx, (x_batch, y_batch, sp_batch) in enumerate(train_loader):
        x_batch = x_batch.long()
        sp_batch = sp_batch.long()
        x_onehot = one_hot_encode_x(x_batch)
        sp_onehot = one_hot_encode_species(sp_batch)

        # Concatenate
        x_input = torch.cat([x_onehot, sp_onehot], dim=1)  # [B, input_dim]

        # Forward pass
        logits = model(x_input)
        loss = criterion(logits, y_batch.float().squeeze())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"🔁 Batch {batch_idx+1} | Loss: {loss.item():.4f}")
