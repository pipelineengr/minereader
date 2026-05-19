# drift_transformer.py
# A transformer encoder that monitors model performance over time
# and predicts whether a deployed GNN is drifting and needs retraining.
#
# WHY A TRANSFORMER HERE?
# After deployment, a mine's grade model is evaluated at regular intervals
# (e.g., after each mine visit). Each evaluation produces a snapshot:
# (MAE, R², timestamp). A sequence of these snapshots over time tells a story —
# is performance stable, gradually degrading, or suddenly dropping?
#
# A transformer is the right architecture because:
# - It processes the ENTIRE sequence at once, not just the last few snapshots
# - Positional encoding preserves the temporal ORDER of visits
# - Self-attention lets the model learn that a spike in MAE at visit 3
#   followed by recovery is different from a monotonic degradation
#
# This maps directly to Stratum's responsibility:
# "Tracking model performance of deployed models over time"

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        """
        Injects information about the position (time order) of each snapshot
        into the sequence. Without this, the transformer treats visit 1 and
        visit 10 as interchangeable — which defeats the purpose for drift detection.

        Uses the classic sinusoidal encoding from "Attention Is All You Need":
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Different frequencies encode different positional scales —
        like how a clock has second, minute, and hour hands.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the positional encoding matrix once and register as a buffer.
        # Buffers are not model parameters (not updated by optimizer),
        # but they ARE saved with the model and moved to GPU with .to(device).
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # position shape: [max_len, 1]

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # div_term: scaling factors for each frequency

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices: sine
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices: cosine
        # pe shape: [max_len, d_model]

        pe = pe.unsqueeze(0)
        # Add batch dimension: [1, max_len, d_model]
        # The 1 broadcasts across any batch size

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: sequence tensor, shape [batch, seq_len, d_model]
        Returns:
            x + positional encoding, same shape
        """
        x = x + self.pe[:, :x.size(1), :]
        # Slice pe to match actual sequence length (may be shorter than max_len)
        return self.dropout(x)


class DriftTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,      # number of metrics per snapshot (MAE + R²)
        d_model: int = 32,       # internal representation size
        nhead: int = 4,          # number of attention heads
        num_layers: int = 2,     # number of transformer encoder layers
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim:  metrics per timestep — 2 for (MAE, R²)
            d_model:    must be divisible by nhead
                        32 / 4 = 8 dimensions per attention head — lightweight but real
            nhead:      multi-head attention splits d_model into nhead parallel attention
                        computations, each attending to different aspects of the sequence
            num_layers: 2 encoder layers is enough — the sequence is short (< 20 visits)
                        and the pattern (drift vs stable) is not deeply hierarchical
        """
        super().__init__()

        # Project raw metrics (2 values) up to d_model dimensions.
        # The transformer needs a richer representation than just (MAE, R²) —
        # this linear projection gives it room to represent patterns.
        self.input_projection = nn.Linear(input_dim, d_model)

        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # TransformerEncoderLayer: one full block of multi-head self-attention + FFN
        # batch_first=True means input shape is [batch, seq_len, d_model]
        # (PyTorch default is [seq_len, batch, d_model] — batch_first is cleaner)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # standard practice: FFN is 4x d_model
            dropout=dropout,
            batch_first=True,
        )

        # Stack num_layers encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Classification head: maps the final sequence representation to a drift score.
        # We use the LAST timestep's representation as the "current state" summary —
        # after attending to the full history, the last position knows whether drift is occurring.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
            # Sigmoid squashes output to [0, 1] — this is the drift risk score.
            # > 0.5 = model is likely drifting, needs retraining
            # < 0.5 = model performance is stable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: sequence of evaluation snapshots, shape [batch, seq_len, input_dim]
               e.g., [32, 10, 2] — batch of 32 sequences, 10 mine visits, (MAE, R²) each

        Returns:
            drift_score: shape [batch, 1] — probability of drift requiring retraining
        """
        # print(f"DriftTransformer input: {x.shape}")

        x = self.input_projection(x)
        # shape: [batch, seq_len, d_model]

        x = self.positional_encoding(x)
        # shape: [batch, seq_len, d_model] — now with temporal position information

        x = self.transformer_encoder(x)
        # shape: [batch, seq_len, d_model]
        # Each position now contains information from the entire sequence
        # via self-attention — visit 3 "knows about" visit 8's performance

        x = x[:, -1, :]
        # Take the LAST timestep: shape [batch, d_model]
        # This is the model's summary of "where are we now given all history"

        drift_score = self.classifier(x)
        # shape: [batch, 1]

        print(f"DriftTransformer output: {drift_score.shape}")
        return drift_score