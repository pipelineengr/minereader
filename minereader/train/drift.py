# train_drift.py
# Generates synthetic drift sequences and trains the DriftTransformer.
#
# THE SIMULATION STRATEGY:
# We don't have real multi-year deployment logs, so we simulate them.
# A "mine visit" produces an evaluation snapshot (MAE, R²).
# We generate two types of sequences:
#   STABLE:  MAE/R² fluctuates slightly around a baseline (normal noise)
#   DRIFTING: MAE degrades monotonically + R² drops over the sequence
#             with added noise to make the pattern non-trivial to detect
#
# The transformer must learn to distinguish these two patterns.
# The key design choice: drift is GRADUAL, not sudden — a sudden change
# is trivial to detect with a threshold rule. Gradual drift requires
# learning the temporal pattern, which is what justifies the transformer.

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from config import RUNS_DIR, LEARNING_RATE
from models.transformer import DriftTransformer


def generate_sequences(
    n_sequences: int = 2000,
    seq_len: int = 12,
    seed: int = 42
):
    """
    Generate synthetic evaluation sequences for drift/stable classification.

    Each sequence represents ~12 mine visits (roughly 6 years of biannual visits).
    Each visit produces (MAE, R²) — normalized to [0,1] range for stable training.

    Returns:
        X: shape [n_sequences, seq_len, 2] — sequences of (MAE, R²) snapshots
        y: shape [n_sequences, 1] — 1 = drifting, 0 = stable
    """
    np.random.seed(seed)
    sequences = []
    labels = []

    for i in range(n_sequences):
        is_drifting = i % 2 == 0  # alternate stable/drifting for balanced classes

        if not is_drifting:
            # STABLE sequence: performance fluctuates around a good baseline
            # MAE stays low (~0.2), R² stays high (~0.85)
            base_mae = np.random.uniform(0.15, 0.25)
            base_r2 = np.random.uniform(0.80, 0.90)
            mae_seq = base_mae + np.random.normal(0, 0.02, seq_len)
            r2_seq = base_r2 + np.random.normal(0, 0.02, seq_len)
            # Small gaussian noise simulates natural variation between visits
        else:
            # DRIFTING sequence: MAE gradually increases, R² gradually decreases
            # Drift starts subtly — the first few visits look stable
            start_mae = np.random.uniform(0.15, 0.25)
            drift_rate = np.random.uniform(0.03, 0.07)  # how fast drift accelerates
            # Drift is not perfectly monotonic — add noise so simple threshold rules fail
            mae_seq = (start_mae +
                      np.arange(seq_len) * drift_rate +
                      np.random.normal(0, 0.02, seq_len))
            r2_seq = (np.random.uniform(0.80, 0.90) -
                     np.arange(seq_len) * (drift_rate * 0.5) +
                     np.random.normal(0, 0.02, seq_len))

        # Clip to valid ranges: MAE ≥ 0, R² ∈ [0, 1]
        mae_seq = np.clip(mae_seq, 0, 1)
        r2_seq = np.clip(r2_seq, 0, 1)

        # Stack into [seq_len, 2] — each row is one mine visit snapshot
        seq = np.stack([mae_seq, r2_seq], axis=1)
        sequences.append(seq)
        labels.append(int(is_drifting))

    X = np.array(sequences, dtype=np.float32)  # [n, seq_len, 2]
    y = np.array(labels, dtype=np.float32).reshape(-1, 1)  # [n, 1]

    # print(f"Generated {n_sequences} sequences: {y.sum():.0f} drifting, {(1-y).sum():.0f} stable")
    return X, y


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training drift transformer on: {device}")

    # --- Generate data ---
    X, y = generate_sequences(n_sequences=2000, seq_len=12)
    print(f"Sequence shape: {X.shape}, Labels shape: {y.shape}")

    # --- Train/val split ---
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train = torch.tensor(X_train).to(device)
    X_val = torch.tensor(X_val).to(device)
    y_train = torch.tensor(y_train).to(device)
    y_val = torch.tensor(y_val).to(device)

    # --- Model ---
    model = DriftTransformer(
        input_dim=2,    # (MAE, R²) per visit
        d_model=32,     # lightweight — the sequence is short, not much capacity needed
        nhead=4,
        num_layers=2,
    ).to(device)
    print(f"DriftTransformer parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCELoss()
    # BCELoss (Binary Cross Entropy) is the right loss for binary classification.
    # It penalizes confident wrong predictions heavily:
    # predicting drift=0.99 when label=0 contributes much more loss than predicting 0.6.

    # --- Training loop ---
    os.makedirs(os.path.join(RUNS_DIR, "drift_transformer"), exist_ok=True)
    best_val_loss = float("inf")
    BATCH_SIZE = 64
    EPOCHS = 50  # short sequences + simple patterns converge fast
    log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Mini-batch training — shuffle each epoch
        perm = torch.randperm(len(X_train))
        train_losses = []

        for i in range(0, len(X_train), BATCH_SIZE):
            batch_idx = perm[i:i + BATCH_SIZE]
            x_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            pred = model(x_batch)
            # pred shape: [batch, 1] — drift probability per sequence
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

            # Accuracy: drift_score > 0.5 → predicted drifting
            val_acc = ((val_pred > 0.5).float() == y_val).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(RUNS_DIR, "drift_transformer", "best.pt"))

        log.append({
            "epoch": epoch,
            "train_loss": np.mean(train_losses),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train_loss: {np.mean(train_losses):.4f} | "
                  f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.3f}")

    pd.DataFrame(log).to_csv(
        os.path.join(RUNS_DIR, "drift_transformer", "train_log.csv"), index=False
    )
    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Saved to runs/drift_transformer/")


if __name__ == "__main__":
    # python minereader/train/drift.py
    train()