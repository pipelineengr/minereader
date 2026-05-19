# train_grade.py
# End-to-end training of CNN encoder + GNN on the Marvin block model.
#
# TRAINING STRATEGY:
# - The GNN trains on the full graph but only supervises on TRAIN nodes.
#   This is called "transductive learning" — the model sees all node positions
#   (including test nodes) during message passing, but only updates weights
#   based on training node losses. This is standard practice in GNNs.
# - CNN is not used yet in this file — that comes in the combined pipeline below.
#   For Day 2 we first confirm the GNN alone works, then add CNN on top.

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
    # This is the PyG Data object saved by prepare_graph.py
    data_path = os.path.join(PROCESSED_DATA_DIR, "marvin.pt")
    data = torch.load(data_path, weights_only=False)
        
    MAX_TRAIN_NODES = 50000
    if data.num_nodes > MAX_TRAIN_NODES:
        perm = torch.randperm(data.num_nodes)[:MAX_TRAIN_NODES]
        data = data.subgraph(perm)
        print(f"Subsampled graph to {data.num_nodes} nodes for training")
    
    # Moving data to GPU: all tensors (x, edge_index, y, pos) are transferred to VRAM.
    # Ran Locally on 4090 Mobile (16GB VRAM)

    data = data.to(device)
    print(f"Loaded Marvin graph: {data.num_nodes} nodes, {data.x.shape[1]} features")

    # --- Train/test split ---
    # We split by NODE INDEX, not by graph — this is a single large graph.
    # train_mask and test_mask are boolean tensors of shape [num_nodes].
    # True = this node is used for supervision during training/evaluation.
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(indices, train_size=TRAIN_SPLIT, random_state=42)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    # print(f"Train nodes: {train_mask.sum()}, Test nodes: {test_mask.sum()}")

    # --- Initialise model ---
    in_channels = data.x.shape[1]  # 5 for Marvin (x, y, z, density, copper)
    model = GradeGNN(in_channels=in_channels, hidden_dim=64).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    # MSELoss penalizes large prediction errors more than small ones.
    # Since grades are normalized, a loss of 1.0 means predictions are
    # off by 1 standard deviation of grade — a useful mental benchmark.

    # LR scheduler: halves learning rate if val loss doesn't improve for 10 epochs.
    # Prevents overshooting the loss minimum in later training stages.
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
        # model.train() enables dropout — important to call before forward pass
        optimizer.zero_grad()
        # Zero gradients from previous step — PyTorch accumulates by default

        out = model(data.x, data.edge_index)
        # out shape: [num_nodes, 1] — a grade prediction for every block

        loss = loss_fn(out[train_mask], data.y[train_mask])
        # Only compute loss on training nodes — test nodes are held out
        loss.backward()
        # Backpropagation: compute gradients of loss w.r.t. all model parameters

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Gradient clipping prevents exploding gradients — a common issue in GNNs.
        # Clips any gradient whose norm exceeds 1.0 back to 1.0.

        optimizer.step()
        # Update weights using computed gradients

        # VALIDATION STEP
        model.eval()
        # model.eval() disables dropout — deterministic predictions for evaluation
        with torch.no_grad():
            # torch.no_grad() disables gradient tracking — faster and less memory
            val_out = model(data.x, data.edge_index)
            val_loss = loss_fn(val_out[test_mask], data.y[test_mask])

            # MAE in normalized space
            val_mae_norm = torch.mean(torch.abs(val_out[test_mask] - data.y[test_mask])).item()

            # MAE in real grade units (inverse-transform)
            # Multiply by grade_std and add grade_mean to convert back to g/t
            val_mae_real = val_mae_norm * data.grade_std
            # print(f"Epoch {epoch}: val_mae_real = {val_mae_real:.4f} g/t")

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(RUNS_DIR, "marvin_grade", "best.pt"))
            # state_dict() saves only the weights, not the model architecture.
            # To load: model = GradeGNN(...); model.load_state_dict(torch.load("best.pt"))

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

    # --- Save final model and training log ---
    torch.save(model.state_dict(),
               os.path.join(RUNS_DIR, "marvin_grade", "last.pt"))

    log_df = pd.DataFrame(log)
    log_df.to_csv(os.path.join(RUNS_DIR, "marvin_grade", "train_log.csv"), index=False)
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Logs saved to runs/marvin_grade/train_log.csv")

    # --- Kriging baseline comparison ---
    print("\nRunning kriging baseline...")
    run_kriging_baseline(data, train_mask, test_mask, device)

    return model, data, test_mask


def run_kriging_baseline(data, train_mask, test_mask, device):
    """
    Ordinary kriging baseline using pykrige.
    Kriging is the industry standard geostatistical interpolation method.
    If our GNN can't beat kriging, the deep learning approach isn't justified.
    This is the benchmark every mining engineer will ask about.
    """
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError:
        print("pykrige not installed. Run: pip install pykrige")
        return

    # Move data to CPU for pykrige (numpy-based, no GPU support)
    pos = data.pos.cpu().numpy()
    y = data.y.cpu().numpy().flatten()

    train_pos = pos[train_mask.cpu().numpy()]
    test_pos = pos[test_mask.cpu().numpy()]
    train_y = y[train_mask.cpu().numpy()]
    test_y = y[test_mask.cpu().numpy()]

    KRIGING_SUBSAMPLE = 50
    if len(train_y) > KRIGING_SUBSAMPLE:
        idx = np.random.choice(len(train_y), KRIGING_SUBSAMPLE, replace=False)
        train_pos_k = train_pos[idx]
        train_y_k = train_y[idx]
        print(f"Kriging: subsampled training set to {KRIGING_SUBSAMPLE} points "
              f"(from {len(train_y)}) for variogram fitting")
    else:
        train_pos_k = train_pos
        train_y_k = train_y

    # Also subsample test points — evaluating kriging at 200k+ locations is slow
    KRIGING_TEST_SAMPLE = 20
    if len(test_y) > KRIGING_TEST_SAMPLE:
        test_idx = np.random.choice(len(test_y), KRIGING_TEST_SAMPLE, replace=False)
        test_pos_k = test_pos[test_idx]
        test_y_k = test_y[test_idx]
        print(f"Kriging: subsampled test set to {KRIGING_TEST_SAMPLE} points")
    else:
        test_pos_k = test_pos
        test_y_k = test_y

    print("Fitting variogram model (this may take 1-2 minutes)...")

    OK = OrdinaryKriging(
        train_pos[:, 0],  # x coordinates
        train_pos[:, 1],  # y coordinates
        train_y,
        variogram_model="spherical",
        verbose=False,
        nlags=6,
        enable_plotting=False,
    )

    # Predict at test locations
    z_pred, _ = OK.execute("points", test_pos[:, 0], test_pos[:, 1])
    kriging_mae = np.mean(np.abs(z_pred - test_y))
    # Convert back to real grade units
    kriging_mae_real = kriging_mae * data.grade_std

    print(f"Kriging baseline MAE: {kriging_mae_real:.4f} g/t "
          f"(evaluated on {KRIGING_TEST_SAMPLE} test points)")
    print(f"(Compare against GNN val_MAE from train_log.csv)")

    # Save kriging result for README comparison table
    result = {
        "kriging_mae_real": float(kriging_mae_real),
        "kriging_train_subsample": KRIGING_SUBSAMPLE,
        "kriging_test_subsample": KRIGING_TEST_SAMPLE,
    }
    with open(os.path.join(RUNS_DIR, "marvin_grade", "kriging_baseline.json"), "w") as f:
        json.dump(result, f)
    print(f"Kriging result saved to runs/marvin_grade/kriging_baseline.json")


if __name__ == "__main__":
    # python minereader/train/train_grade_krigging.py
    train()