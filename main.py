import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
#from esm.models.esmc import ESMC
#from esm.sdk.api import ESMProtein, LogitsConfig
#from modeling import OuterProductModule, CrossAttentionModule, CNN
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


with open('wild_type.json', 'r') as file:
    wild_type = json.load(file)
    wild_type = wild_type['wild_type']

with open('updated_species_dict.json', 'r') as file:
    species_dict = json.load(file)

with open('closest_species.json', 'r') as file:
    closest_species = json.load(file)

def _annotate_bars(ax, fmt="{:,.0f}", padding=3):
    """Annotate bar heights above each bar in the current Axes."""
    for p in ax.patches:
        height = p.get_height()
        if height is None or np.isnan(height):
            continue
        x = p.get_x() + p.get_width() / 2.0
        y = p.get_y() + height
        ax.text(x, y + (height * 0.01) + padding * 0.0, fmt.format(height),
                ha='center', va='bottom', fontsize=9, rotation=0)

def plot_rbd_dataset_summaries(df: pd.DataFrame):
    """
    Expects a DataFrame with columns:
      species columns: each ∈ {1, 0, -1}
      'ed' : integer edit distance from wild type
      'varied_group' : categorical
    """

    # --- Create output folder ---
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # --- Identify species columns ---
    species_cols = [
        'mouse', 'ihbat', 'cattle', 'human', 'willow', 'tit', 'ostrich', 'dog',
        'cat', 'monkey', 'horse', 'owl', 'boar', 'mink', 'possum', 'pangolin',
        'chicken', 'human_new', 'civet', 'rat'
    ]
    species_cols = [c for c in species_cols if c in df.columns]

    # --- 1) Distribution of labels per species (excluding -1) ---
    long = df[species_cols].melt(var_name='species', value_name='label')
    long = long[long['label'].isin([0, 1])]  # exclude -1 (missing)

    counts = (
        long.groupby(['species', 'label'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)  # show 0 then 1
    )

    label_name_map = {0: 'no bind (0)', 1: 'bind (1)'}

    fig1, ax1 = plt.subplots(figsize=(max(10, len(species_cols) * 0.75), 6))
    x = np.arange(len(counts.index))
    width = 0.35

    # Grouped bars: 0 and 1
    ax1.bar(x - width/2, counts[0].values, width, label=label_name_map[0])
    ax1.bar(x + width/2, counts[1].values, width, label=label_name_map[1])

    ax1.set_title('Label distribution per species (excluding missing)')
    ax1.set_xlabel('Species')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(counts.index, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    _annotate_bars(ax1)  # add counts above bars

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'labels_per_species.png'), dpi=150)

    # --- 2) Edit distance distribution (histogram) with counts on bins ---
    if 'ed' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ed_vals = pd.to_numeric(df['ed'], errors='coerce').dropna()

        if len(ed_vals) > 0:
            ed_min, ed_max = int(ed_vals.min()), int(ed_vals.max())
            if ed_max - ed_min <= 60:
                bins = np.arange(ed_min, ed_max + 2) - 0.5  # integer-centered bins
                n, b, patches = ax2.hist(ed_vals, bins=bins)
                ax2.set_xticks(np.arange(ed_min, ed_max + 1))
            else:
                n, b, patches = ax2.hist(ed_vals, bins='auto')

            ax2.set_title('Edit distance (ed) distribution')
            ax2.set_xlabel('Edit distance from wild type')
            ax2.set_ylabel('Count')
            ax2.grid(axis='y', linestyle='--', alpha=0.5)

            # Annotate each bin with its count
            for rect, count in zip(patches, n):
                if count <= 0:
                    continue
                x = rect.get_x() + rect.get_width() / 2.0
                y = rect.get_height()
                ax2.text(x, y, f"{int(count)}", ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'edit_distance_distribution.png'), dpi=150)
        else:
            print("Column 'ed' has no numeric values.")
    else:
        print("Column 'ed' not found; skipping edit distance plot.")

    # --- 3) Varied group value counts (with annotations) ---
    if 'varied_group' in df.columns:
        vc = df['varied_group'].astype('category').value_counts(dropna=False)

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.bar(vc.index.astype(str), vc.values)
        ax3.set_title('Value counts: varied_group')
        ax3.set_xlabel('varied_group')
        ax3.set_ylabel('Count')
        ax3.set_xticklabels(vc.index.astype(str), rotation=45, ha='right')
        ax3.grid(axis='y', linestyle='--', alpha=0.5)
        _annotate_bars(ax3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'varied_group_value_counts.png'), dpi=150)
    else:
        print("Column 'varied_group' not found; skipping varied_group plot.")


def plot_split_distributions(df: pd.DataFrame,
                             split_cols=('random_split', 'single_vs_multiple_split','single_only_split')):
    """
    Plots value-count distributions (with bar annotations) for each split column.
    Saves one PNG per split column under 'plots/'.

    Works with categorical or numeric splits. NaNs are shown as a category.
    """
    output_dir = _ensure_output_dir("plots")

    for col in split_cols:
        if col not in df.columns:
            print(f"Column '{col}' not found; skipping.")
            continue

        # Build value counts (keep NaN as explicit category)
        vc = df[col].astype('category').value_counts(dropna=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(vc.index.astype(str), vc.values)
        ax.set_title(f"Value counts: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.set_xticklabels(vc.index.astype(str), rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        _annotate_bars(ax)

        plt.tight_layout()
        fname = f"{col}_value_counts.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150)


def _ensure_output_dir(output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
# Usage example:

def plot_no_labels_count(df, prefer_col='no_labels'):
    """
    Plots the distribution of the count of known labels per sequence
    (i.e., how many species are not -1). Saves a labeled bar chart to plots/.

    If the expected column 'no_labels' is missing but 'no_layers' exists,
    it will use that instead.
    """
    output_dir = _ensure_output_dir("plots")

    # Pick the column to use
    col = prefer_col
    if col not in df.columns:
        if 'no_layers' in df.columns:
            col = 'no_layers'
            print("Using 'no_layers' since 'no_labels' was not found.")
        else:
            print("Neither 'no_labels' nor 'no_layers' found; skipping plot.")
            return

    # Clean and count discrete values
    vals = pd.to_numeric(df[col], errors='coerce').dropna().astype(int)
    if len(vals) == 0:
        print(f"Column '{col}' has no numeric values; skipping plot.")
        return

    counts = vals.value_counts().sort_index()  # index = number of known labels, value = frequency

    # Plot as a bar chart (one bar per count value)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"Distribution of {col} (known labels per sequence)")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_xticklabels(counts.index.astype(str), rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    _annotate_bars(ax)

    plt.tight_layout()
    fname = f"{col}_distribution.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)

def plot_varied_counts_for_no_labels(
    df,
    k: int,
    no_labels_col: str = 'no_labels',
    is_varied_col: str = 'is_varied'
):
    """
    Filter rows where no_labels == k and plot value counts of is_varied.
    Saves to plots/varied_counts_no_labels_{k}.png.

    - If is_varied is missing, it's computed from species labels (ignoring -1).
    - If 'no_labels' is missing but 'no_layers' exists, that is used instead.
    """
    output_dir = _ensure_output_dir("plots")

    # Choose the no_labels column (fallback to 'no_layers' if needed)
    if no_labels_col not in df.columns:
        if 'no_layers' in df.columns:
            no_labels_col = 'no_layers'
            print("Using 'no_layers' since 'no_labels' was not found.")
        else:
            print("Neither 'no_labels' nor 'no_layers' found; skipping plot.")
            return

    # Ensure is_varied exists (boolean)
    df = _compute_is_varied_if_missing(df, is_varied_col=is_varied_col)
    is_varied = df[is_varied_col].astype(bool)

    # Clean/align no_labels and filter to the requested k
    k_series = pd.to_numeric(df[no_labels_col], errors='coerce').astype('Int64')
    df_k = df[k_series == k].copy()
    if df_k.empty:
        print(f"No rows with {no_labels_col} == {k}; skipping.")
        return

    # Count varied vs non-varied
    counts = df_k[is_varied_col].astype(bool).value_counts()
    non_varied = int(counts.get(False, 0))
    varied = int(counts.get(True, 0))

    # Build a tiny bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ['non-varied', 'varied']
    values = [non_varied, varied]
    ax.bar(categories, values)
    ax.set_title(f"Varied counts for {no_labels_col} = {k}")
    ax.set_xlabel('is_varied')
    ax.set_ylabel('Count')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    _annotate_bars(ax)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"varied_counts_no_labels_{k}.png")
    plt.savefig(out_path, dpi=150)

    print(f"Saved: {out_path}")


def _compute_is_varied_if_missing(df: pd.DataFrame, is_varied_col='is_varied'):
    """
    Define is_varied as: in the row, among species labels (ignoring -1),
    there exists at least one 0 and at least one 1.
    """
    if is_varied_col in df.columns:
        # Normalize to boolean
        iv = df[is_varied_col]
        if iv.dtype != bool:
            # Accept 0/1 or strings; convert to bool
            df[is_varied_col] = df[is_varied_col].astype(int).astype(bool)
        return df

    species_cols = _species_cols_in_df(df)
    if not species_cols:
        raise ValueError("Cannot compute is_varied: no species columns found.")

    # Vectorized computation:
    # mask_known: values that are 0 or 1
    vals = df[species_cols]
    mask0 = (vals == 0)
    mask1 = (vals == 1)
    has0 = mask0.any(axis=1)
    has1 = mask1.any(axis=1)
    df[is_varied_col] = (has0 & has1)
    return df

def _species_cols_in_df(df: pd.DataFrame):
    # Your species list (kept here so we can compute is_varied if needed)
    cols = [
        'mouse', 'ihbat', 'cattle', 'human', 'willow', 'tit', 'ostrich', 'dog',
        'cat', 'monkey', 'horse', 'owl', 'boar', 'mink', 'possum', 'pangolin',
        'chicken', 'human_new', 'civet', 'rat'
    ]
    return [c for c in cols if c in df.columns]


def add_single_only_split(df, no_labels_col='no_labels', seed=42):
    """
    Adds a column 'single_only_split' with values 'train', 'valid', 'test'
    ONLY for rows where no_labels == 1.
    Distribution: 70% train, 15% valid, 15% test.
    Other rows get NaN.
    """
    rng = np.random.default_rng(seed)

    # Initialize with NaN
    df['single_only_split'] = np.nan

    # Identify rows with exactly 1 known label
    mask_single = df[no_labels_col] == 1
    idx_single = df[mask_single].index
    n = len(idx_single)

    if n == 0:
        print("No rows with no_labels == 1; no split applied.")
        return df

    # Shuffle indices
    shuffled_idx = rng.permutation(idx_single)

    # Calculate split sizes
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)
    # The remainder goes to test
    n_test = n - n_train - n_valid

    # Assign splits
    train_idx = shuffled_idx[:n_train]
    valid_idx = shuffled_idx[n_train:n_train + n_valid]
    test_idx = shuffled_idx[n_train + n_valid:]

    df.loc[train_idx, 'single_only_split'] = 'train'
    df.loc[valid_idx, 'single_only_split'] = 'valid'
    df.loc[test_idx, 'single_only_split'] = 'test'

    return df

if __name__ == "__main__":
    data = pd.read_csv('mls_data_full.csv')
    print(data.shape)
    print(data.columns)
    print(len(data['aa_seq'].values[0]))
    add_single_only_split(data)
    plot_rbd_dataset_summaries(data)
    plot_split_distributions(data)
    plot_no_labels_count(data)
    for k in range(1,20):
        plot_varied_counts_for_no_labels(data, k)

    print(data.columns)