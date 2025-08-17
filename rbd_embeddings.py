import torch
from typing import Callable, Literal
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import time
from tqdm import tqdm
from pathlib import Path
# Amino acid vocabulary
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
index_to_aa = {i: aa for i, aa in enumerate(AMINO_ACIDS)}
unk_index = len(AMINO_ACIDS)


def decode_fn(indices: list[int]) -> str:
    """Convert list of indices to amino acid string."""
    return "".join([index_to_aa[i] for i in indices if i in index_to_aa])


def embed_sequences(
    categorical_seqs: torch.Tensor,
    model_name: str = "esmc_300m",
    device: Literal["cpu", "cuda"] = "cpu",
    dtype: Literal["float32", "float64"] = "float32",
) -> torch.Tensor:
    """
    Embed categorically encoded protein sequences using ESMC model from Meta's esm SDK.

    Args:
        categorical_seqs (torch.Tensor): Tensor of shape (N, L) with categorical indices.
        model_name (str): Name of the ESMC model (e.g., "esmc_300m").
        device (str): "cpu" or "cuda".
        dtype (str): Output tensor dtype ("float32" or "float64").

    Returns:
        torch.Tensor: Tensor of shape (N, D) with embedded sequences.
    """
    assert dtype in ["float32", "float64"], "dtype must be 'float32' or 'float64'"
    assert device in ["cpu", "cuda"], "device must be 'cpu' or 'cuda'"

    # Load ESMC model and move to device
    client = ESMC.from_pretrained(model_name).to(device)

    embeddings = []
    for seq_tensor in categorical_seqs:
        seq_str = decode_fn(seq_tensor.tolist())
        protein = ESMProtein(sequence=seq_str)
        protein_tensor = client.encode(protein)

        logits_output = client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        emb = logits_output.embeddings  # Tensor [1, D]
        embeddings.append(emb.squeeze(0))  # Shape: [D]

    result = torch.stack(embeddings)  # Shape: [N, D]

    # Convert dtype
    result = result.to(torch.float64 if dtype == "float64" else torch.float32)

    return result.to(device)

if __name__ == "__main__":
    LOG_FILE = "embedding_log.txt"
    NUM_SEQUENCES = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = "float32"
    MODEL_NAME = "esmc_300m"
    OUTPUT_FILE = "x_embeddings_100.pt"
    # ===================================

    # Initialize log
    log_path = Path(LOG_FILE)
    log_path.write_text("🧬 ESMC Embedding Log\n\n")

    def log(msg):
        with open(LOG_FILE, "a") as f:
            f.write(f"{msg}\n")

    # Load sequences
    x_categorical = torch.load("x_categorical.pt")
    x_categorical = x_categorical[:NUM_SEQUENCES]
    num_seqs = x_categorical.shape[0]
    log(f"Loaded {num_seqs} sequences.\n")

    # Start timing
    start_time = time.time()

    # Embed with per-sequence tracking
    log(f"Using model: {MODEL_NAME} | device: {DEVICE} | dtype: {DTYPE}")
    log("Embedding sequences...\n")

    embeddings = []
    for i in tqdm(range(num_seqs), desc="Embedding sequences", unit="seq"):
        seq_tensor = x_categorical[i].unsqueeze(0)  # Shape: [1, L]
        start_seq_time = time.time()

        emb = embed_sequences(
            categorical_seqs=seq_tensor,
            model_name=MODEL_NAME,
            device=DEVICE,
            dtype=DTYPE,
        )
        embeddings.append(emb[0])  # Append [D] vector

        elapsed_seq = time.time() - start_seq_time
        log(f"[{i+1}/{num_seqs}] Embedded in {elapsed_seq:.4f} seconds")

    # Stack and save
    embedding_tensor = torch.stack(embeddings)
    torch.save(embedding_tensor, OUTPUT_FILE)

    # Final timing
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_seqs

    log(f"\n✅ Embedding complete. Saved to: {OUTPUT_FILE}")
    log(f"⏱️ Total time: {total_time:.2f} seconds")
    log(f"⏱️ Avg time/sequence: {avg_time:.4f} seconds")