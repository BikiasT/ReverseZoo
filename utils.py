import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_label_distribution_per_column(y, subgroup_name, out_dir="plots"):
    """
    Save a bar chart of label distribution (0, 1) per column for the given y.
    Excludes -1 values entirely.

    Args:
        y: torch.Tensor or np.ndarray, shape (n_samples, n_labels)
        subgroup_name: str, name to use in plot title and filename (e.g., 'single_test')
        out_dir: directory to save plots
    """
    # Convert to numpy
    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()
    else:
        y_np = np.asarray(y)

    if y_np.ndim == 1:
        y_np = y_np[:, None]

    n_cols = y_np.shape[1]

    # Only count valid labels (0 and 1)
    counts_0 = (y_np == 0).sum(axis=0)
    counts_1 = (y_np == 1).sum(axis=0)

    indices = np.arange(n_cols)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars0 = ax.bar(indices - width/2, counts_0, width, label='0')
    bars1 = ax.bar(indices + width/2, counts_1, width, label='1')

    ax.set_xlabel("Column Index")
    ax.set_ylabel("Count")
    ax.set_title(f"Label Distribution per Column - {subgroup_name}")
    ax.set_xticks(indices)
    ax.legend()

    # Annotate counts above bars
    for bar in bars0 + bars1:
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # offset above bar
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    # Save plot
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{subgroup_name}_label_distribution.png")
    plt.savefig(save_path)
    plt.close(fig)

    print(f"Plot saved to {save_path}")
