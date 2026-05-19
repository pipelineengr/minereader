# cnn_encoder.py
# A 1D Convolutional Neural Network that processes raw drill core assay sequences.
#
# WHY A CNN HERE?
# Each drill hole produces a sequence of measurements at regular depth intervals
# (e.g., Cu, Fe, S concentrations every 1 metre down the hole).
# A 1D CNN is the right tool for this because:
#   - It's translation-invariant: a mineralizing horizon at 50m looks the same as at 80m
#   - It's efficient: Conv1d slides a small filter across the sequence, learning local patterns
#   - The output embedding captures the geochemical "signature" of that drill hole
#
# The embedding is then used as the node feature vector fed into the GNN.
# Because CNN + GNN train end-to-end, the CNN learns features that actually
# HELP grade prediction, not just features that describe the assay sequence.

import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 32):
        """
        Args:
            in_channels:    number of assay measurements per depth interval
                            (e.g., 1 if only Au, 2 if Au + Cu)
            embedding_dim:  size of the output embedding vector per drill hole
                            this becomes the node feature dimension fed into the GNN
        """
        super().__init__()

        # Three Conv1d layers with increasing channel depth.
        # Conv1d(in_channels, out_channels, kernel_size)
        # kernel_size=3 means each filter looks at 3 consecutive depth intervals at once.
        # This is like a sliding window that detects local geochemical patterns.
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        # padding=1 keeps the sequence length the same after convolution

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        # We double the channels: more filters = more types of patterns detected

        self.conv3 = nn.Conv1d(32, embedding_dim, kernel_size=3, padding=1)
        # Final conv outputs embedding_dim channels — this is the "richness" of the embedding

        self.relu = nn.ReLU()
        # ReLU introduces non-linearity: without it, stacking conv layers is
        # mathematically equivalent to a single linear transformation (useless)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # Global average pool collapses the entire sequence into a single value per channel.
        # Input:  [batch, embedding_dim, sequence_length]
        # Output: [batch, embedding_dim, 1]
        # This is how we get a FIXED-LENGTH embedding regardless of drill hole depth.
        # A 100m hole and a 200m hole both produce an embedding_dim-sized vector.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw assay sequence, shape [batch, in_channels, sequence_length]
               e.g., [32, 1, 50] for a batch of 32 drill holes, 1 assay type, 50 depth intervals

        Returns:
            embedding: shape [batch, embedding_dim]
        """
        # print(f"CNN input shape: {x.shape}")

        x = self.relu(self.conv1(x))
        # shape: [batch, 16, sequence_length]

        x = self.relu(self.conv2(x))
        # shape: [batch, 32, sequence_length]

        x = self.relu(self.conv3(x))
        # shape: [batch, embedding_dim, sequence_length]

        x = self.global_avg_pool(x)
        # shape: [batch, embedding_dim, 1]

        x = x.squeeze(-1)
        # squeeze removes the trailing 1: shape becomes [batch, embedding_dim]
        # print(f"CNN output (embedding) shape: {x.shape}")

        return x