import torch
from typing import Literal
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
    Embed sequences using ESMC and return mean-pooled per-sequence embeddings [N, D]
    """
    assert dtype in ["float32", "float64"], "dtype must be 'float32' or 'float64'"
    assert device in ["cpu", "cuda"], "device must be 'cpu' or 'cuda'"

    # Load ESMC model
    client = ESMC.from_pretrained(model_name).to(device)

    embeddings = []
    for seq_tensor in categorical_seqs:
        seq_str = decode_fn(seq_tensor.tolist())
        protein = ESMProtein(sequence=seq_str)
        protein_tensor = client.encode(protein)

        logits_output = client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        # logits_output.embeddings: [1, L, D]
        emb = logits_output.embeddings.mean(dim=1).squeeze(0)  # [D]
        embeddings.append(emb)

    result = torch.stack(embeddings)  # [N, D]
    result = result.to(torch.float64 if dtype == "float64" else torch.float32)
    return result.to(device)

# ========== Main ==========

if __name__ == "__main__":
    # Configuration
    EMBEDDING_DIR = Path("embeddings")
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    LOG_FILE = EMBEDDING_DIR / "embedding_log.txt"
    OUTPUT_FILE = EMBEDDING_DIR / "x_embeddings_56.pt"

    NUM_SEQUENCES = 56
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = "float32"
    MODEL_NAME = "esmc_600m"

    # Initialize log
    LOG_FILE.write_text("🧬 ESMC Embedding Log\n\n")

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
    log(f"Using model: {MODEL_NAME} | device: {DEVICE} | dtype: {DTYPE}")
    log("Embedding sequences...\n")

    embeddings = []
    for i in tqdm(range(num_seqs), desc="Embedding sequences", unit="seq"):
        seq_tensor = x_categorical[i].unsqueeze(0)  # [1, L]
        start_seq_time = time.time()

        emb = embed_sequences(
            categorical_seqs=seq_tensor,
            model_name=MODEL_NAME,
            device=DEVICE,
            dtype=DTYPE,
        )
        embeddings.append(emb[0])  # [D]

        elapsed_seq = time.time() - start_seq_time
        log(f"[{i+1}/{num_seqs}] Embedded in {elapsed_seq:.4f} seconds")

    # Stack and save
    embedding_tensor = torch.stack(embeddings)  # [100, D]
    torch.save(embedding_tensor, OUTPUT_FILE)

    # Final timing
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_seqs

    log(f"\n✅ Embedding complete. Saved to: {OUTPUT_FILE}")
    log(f"⏱️ Total time: {total_time:.2f} seconds")
    log(f"⏱️ Avg time/sequence: {avg_time:.4f} seconds")
