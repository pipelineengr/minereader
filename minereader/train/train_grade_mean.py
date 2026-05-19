# train_grade.py
# End-to-end training of GNN on the Marvin block model.
# Kriging baseline replaced with mean predictor for speed.
# Run once overnight with kriging if you want that number for the README.

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.model_selection import train_test_split
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from config import (DEVICE, PROCESSED_DATA_DIR, RUNS_DIR,
                    LEARNING_RATE, EPOCHS, TRAIN_SPLIT)
from models.gcn import GradeGNN

def train():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- Load processed graph ---
    data_path = os.path.join(PROCESSED_DATA_DIR, "marvin.pt")
    data = torch.load(data_path, weights_only=False)
    print(f"Loaded Marvin graph: {data.num_nodes} nodes, {data.x.shape[1]} features")

    # --- Subsample BEFORE moving to GPU ---
    # Must happen on CPU — subgraph() uses CPU index operations internally
    MAX_TRAIN_NODES = 50000
    if data.num_nodes > MAX_TRAIN_NODES:
        perm = torch.randperm(data.num_nodes)[:MAX_TRAIN_NODES]
        data = data.subgraph(perm)
        print(f"Subsampled graph to {data.num_nodes} nodes")

    # --- Move to GPU ---
    data = data.to(device)

    # --- Train/test split ---
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(indices, train_size=TRAIN_SPLIT, random_state=42)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    print(f"Train nodes: {train_mask.sum()}, Test nodes: {test_mask.sum()}")

    # --- Mean predictor baseline ---
    # Predict the training mean for every test block.
    # This is the absolute floor — any model that can't beat this is useless.
    # The gap between mean predictor MAE and GNN MAE is your headline improvement number.
    train_grades = data.y[train_mask].cpu().numpy()
    test_grades = data.y[test_mask].cpu().numpy()
    mean_pred = train_grades.mean()
    mean_mae_norm = np.mean(np.abs(mean_pred - test_grades))
    mean_mae_real = mean_mae_norm * data.grade_std
    print(f"\nMean predictor baseline MAE: {mean_mae_real:.4f} g/t")
    print(f"(GNN must beat this to be useful)\n")

    # --- Initialise model ---
    in_channels = data.x.shape[1]
    model = GradeGNN(in_channels=in_channels, hidden_dim=64).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    # LR scheduler: halves learning rate if val loss doesn't improve for 10 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=True
    )

    # --- Training loop ---
    os.makedirs(os.path.join(RUNS_DIR, "marvin_grade"), exist_ok=True)
    best_val_loss = float("inf")
    log = []

    for epoch in range(1, EPOCHS + 1):
        # TRAIN STEP
        model.train()
        # model.train() enables dropout
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        # out shape: [num_nodes, 1] — grade prediction for every block

        loss = loss_fn(out[train_mask], data.y[train_mask])
        # Only compute loss on training nodes — test nodes held out
        loss.backward()
        # Backpropagation: compute gradients w.r.t. all model parameters

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Prevents exploding gradients — common issue in GNNs

        optimizer.step()

        # VALIDATION STEP
        model.eval()
        # model.eval() disables dropout for deterministic evaluation
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = loss_fn(val_out[test_mask], data.y[test_mask])
            val_mae_norm = torch.mean(torch.abs(val_out[test_mask] - data.y[test_mask])).item()
            val_mae_real = val_mae_norm * data.grade_std
            # Multiply by grade_std to convert normalized MAE back to real g/t units

        scheduler.step(val_loss)

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(RUNS_DIR, "marvin_grade", "best.pt"))
            # state_dict() saves weights only — load with:
            # model.load_state_dict(torch.load("best.pt"))

        log.append({
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "val_mae_norm": val_mae_norm,
            "val_mae_real": val_mae_real,
        })

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train_loss: {loss.item():.4f} | "
                  f"val_loss: {val_loss.item():.4f} | val_MAE: {val_mae_real:.4f} g/t")

    # --- Save final checkpoint and log ---
    torch.save(model.state_dict(),
               os.path.join(RUNS_DIR, "marvin_grade", "last.pt"))

    log_df = pd.DataFrame(log)
    log_df.to_csv(os.path.join(RUNS_DIR, "marvin_grade", "train_log.csv"), index=False)

    # --- Save baseline comparison ---
    # This is what goes in your README comparison table
    best_gnn_mae = log_df["val_mae_real"].min()
    results = {
        "mean_predictor_mae_real": float(mean_mae_real),
        "gnn_best_mae_real": float(best_gnn_mae),
        "improvement_pct": float((mean_mae_real - best_gnn_mae) / mean_mae_real * 100),
    }
    with open(os.path.join(RUNS_DIR, "marvin_grade", "baseline_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results ===")
    print(f"Mean predictor MAE : {mean_mae_real:.4f} g/t")
    print(f"GNN best MAE       : {best_gnn_mae:.4f} g/t")
    print(f"Improvement        : {results['improvement_pct']:.1f}%")
    print(f"Logs → runs/marvin_grade/train_log.csv")
    print(f"Results → runs/marvin_grade/baseline_comparison.json")


if __name__ == "__main__":
    # python minereader/train/train_grade_mean.py
    train()