import torch
import torch.nn as nn

class SimpleBindingTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        transformer_dim: int,
        num_layers: int,
        num_species: int
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=None)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(embed_dim, num_species)

    def forward(self, x):
        x = self.embedding(x)             # (batch, seq_len, embed_dim)
        x = self.transformer(x)           # (batch, seq_len, embed_dim)
        x = x.mean(dim=1)                 # average pooling over sequence
        return self.classifier(x)         # (batch, num_species)
