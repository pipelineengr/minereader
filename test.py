import torch
import pandas as pd
import sys
sys.path.append(".")
from minereader.models.gcn import GradeGNN

# --- Graph data ---
data = torch.load("data/processed/marvin.pt", weights_only=False)
print("=== Graph Data ===")
print(f"Nodes: {data.num_nodes}")
print(f"Node features: {data.x.shape}  (x, y, z, density, copper)")
print(f"Edges: {data.edge_index.shape[1]}  ({data.edge_index.shape[1] // data.num_nodes} per node)")
print(f"Grade labels: {data.y.shape}")
print(f"Grade mean (real units): {data.grade_mean:.4f}")
print(f"Grade std  (real units): {data.grade_std:.4f}")

# --- Model weights ---
print("\n=== Model Checkpoint ===")
state_dict = torch.load("runs/marvin_grade/best.pt", weights_only=True)
for name, tensor in state_dict.items():
    print(f"{name:40s} {tensor.shape}")

# --- Training log ---
print("\n=== Training Log (last 5 epochs) ===")
log = pd.read_csv("runs/marvin_grade/train_log.csv")
print(log.tail(5).to_string(index=False))
print(f"\nBest val_MAE: {log['val_mae_real'].min():.4f} g/t at epoch {log['val_mae_real'].idxmin() + 1}")