# prepare_graph.py
# Converts a raw block model CSV into a PyTorch Geometric Data object.
# A PyG Data object is the core data structure for graph learning:
#   data.x          — node feature matrix, shape [num_nodes, num_features]
#   data.edge_index — graph connectivity, shape [2, num_edges] (COO format)
#   data.y          — node labels (grade values), shape [num_nodes, 1]
#   data.pos        — raw spatial coordinates, shape [num_nodes, 3]

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import sys

# Allow imports from project root
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from config import DATASET_CONFIGS, KNN_K, DEVICE, PROCESSED_DATA_DIR

def load_and_prepare(dataset_name: str, csv_path: str) -> Data:
    """
    Load a block model CSV and return a PyG Data object ready for GNN training.
    
    Args:
        dataset_name: key in DATASET_CONFIGS — 'marvin' or 'mclaughlin'
        csv_path: path to the raw CSV file
    
    Returns:
        PyG Data object with node features, edges, labels, and coordinates
    """
    cfg = DATASET_CONFIGS[dataset_name]
    
    # --- Load CSV ---
    print(f"[{dataset_name}] Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[{dataset_name}] Raw shape: {df.shape}")
    print(f"[{dataset_name}] Columns: {list(df.columns)}")

    grade_col = cfg["grade"]
    p99 = df[grade_col].quantile(0.99)
    p01 = df[grade_col].quantile(0.01)
    outlier_count = ((df[grade_col] > p99) | (df[grade_col] < p01)).sum()
    print(f"[{dataset_name}] Clipping {outlier_count} outliers outside [{p01:.4f}, {p99:.4f}]")
    df[grade_col] = df[grade_col].clip(lower=p01, upper=p99)

    # --- Drop rows where the grade target is missing ---
    # In real block models, some blocks may have NaN grades (unsampled blocks).
    # These can't be used as training labels, but CAN be used as inference targets.
    # For now we drop them; on Day 3 you'll revisit this for the zero-shot experiment.
    initial_len = len(df)
    df = df.dropna(subset=[cfg["grade"]])
    print(f"[{dataset_name}] Dropped {initial_len - len(df)} rows with missing grade. Remaining: {len(df)}")

    # --- Extract and normalize spatial coordinates ---
    # Normalization is critical: Marvin uses local block-unit coordinates (~0–100 range)
    # while McLaughlin uses absolute UTM coordinates (~thousands range).
    # Without normalization, the KNN distance thresholds would be incomparable across mines.
    coords = df[[cfg["x"], cfg["y"], cfg["z"]]].values.astype(np.float32)
    scaler_coords = StandardScaler()
    coords_norm = scaler_coords.fit_transform(coords)
    # print(f"[{dataset_name}] Coord stats after norm — mean: {coords_norm.mean(axis=0)}, std: {coords_norm.std(axis=0)}")

    # --- Build node feature matrix ---
    # Node features are what each block "knows about itself" before seeing its neighbours.
    # The GNN will then aggregate neighbour features during message passing.
    # We include: normalized x, y, z, density.
    # Grade is the LABEL (data.y), not a feature — including it would be data leakage.
    features = [coords_norm]  # start with spatial coords as features

    if cfg["density"] in df.columns:
        density = df[[cfg["density"]]].values.astype(np.float32)
        scaler_density = StandardScaler()
        density_norm = scaler_density.fit_transform(density)
        features.append(density_norm)
        # print(f"[{dataset_name}] Density feature added, shape: {density_norm.shape}")
    else:
        print(f"[{dataset_name}] WARNING: density column '{cfg['density']}' not found, skipping")

    if cfg.get("aux_grade") and cfg["aux_grade"] in df.columns:
        aux = df[[cfg["aux_grade"]]].values.astype(np.float32)
        scaler_aux = StandardScaler()
        features.append(scaler_aux.fit_transform(aux))
        # print(f"[{dataset_name}] aux_grade '{cfg['aux_grade']}' added as node feature")

    # Stack all features into a single matrix: shape [num_nodes, num_features]
    x = np.hstack(features)
    print(f"[{dataset_name}] Node feature matrix shape: {x.shape}")
    # Each row is one block. Each column is one feature (x, y, z, density).

    # --- Extract grade labels ---
    y = df[[cfg["grade"]]].values.astype(np.float32)
    scaler_grade = StandardScaler()
    y_norm = scaler_grade.fit_transform(y)
    # We normalize grades too so MSELoss operates on unit-scale values.
    # Remember to inverse_transform predictions before computing real-world MAE.
    print(f"[{dataset_name}] Grade label shape: {y_norm.shape}, mean: {y_norm.mean():.4f}")

    # --- Convert to tensors ---
    x_tensor = torch.tensor(x, dtype=torch.float)
    y_tensor = torch.tensor(y_norm, dtype=torch.float)
    pos_tensor = torch.tensor(coords_norm, dtype=torch.float)
    # pos is stored separately so we can use it for visualization later (geologist report)

    # --- Build spatial KNN graph ---
    # knn_graph connects each node to its K nearest neighbours based on pos coordinates.
    # This is what makes it a GRAPH rather than a flat feature matrix —
    # the edge_index encodes which blocks are spatially adjacent.
    # edge_index shape: [2, num_edges] — each column is a directed edge (source → target)
    print(f"[{dataset_name}] Building KNN graph with K={KNN_K}...")
    edge_index = knn_graph(pos_tensor, k=KNN_K, loop=False)
    # loop=False means a block doesn't connect to itself
    print(f"[{dataset_name}] edge_index shape: {edge_index.shape}")
    # Expected: [2, num_nodes * KNN_K] — each node has exactly K outgoing edges

    # --- Assemble PyG Data object ---
    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor,
        pos=pos_tensor,
        num_nodes=x_tensor.shape[0],
    )
    # Attach metadata for later use (e.g., inverse scaling in the API)
    data.dataset_name = dataset_name
    data.grade_mean = float(scaler_grade.mean_[0])
    data.grade_std = float(scaler_grade.scale_[0])

    print(f"[{dataset_name}] Final Data object: {data}")
    # Expected output: Data(x=[N, F], edge_index=[2, E], y=[N, 1], pos=[N, 3])

    # --- Save processed graph ---
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DATA_DIR, f"{dataset_name}.pt")
    torch.save(data, save_path)
    print(f"[{dataset_name}] Saved to {save_path}")

    return data, scaler_grade

if __name__ == "__main__":
    # Run this script directly to process both datasets:
    # python minereader/data/prepare_graph.py

    marvin_data, _ = load_and_prepare(
        "marvin",
        os.path.join("data/raw", "marvin.csv")
    )
    mclaughlin_data, _ = load_and_prepare(
        "mclaughlin",
        os.path.join("data/raw", "mclaughlin.csv")
    )

    print("\n=== Summary ===")
    print(f"Marvin nodes: {marvin_data.num_nodes}, edges: {marvin_data.edge_index.shape[1]}")
    print(f"McLaughlin nodes: {mclaughlin_data.num_nodes}, edges: {mclaughlin_data.edge_index.shape[1]}")

"""Function for visualization, uncomment if needed

import plotly.express as px

def visualize_grade(data, dataset_name, max_points=100000):
    coords = data.pos.numpy()
    grades = data.y.numpy().flatten()
    
    if len(grades) > max_points:
        idx = np.random.choice(len(grades), max_points, replace=False)
        coords = coords[idx]
        grades = grades[idx]
        print(f"[{dataset_name}] Subsampled to {max_points} points for visualization")

    fig = px.scatter_3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        color=grades,
        color_continuous_scale="Viridis",
        title=f"{dataset_name} — Grade Distribution",
        labels={"color": "Grade (normalized)"}
    )
    fig.write_html(f"runs/{dataset_name}_grade_viz.html")
    print(f"Saved visualization to runs/{dataset_name}_grade_viz.html")

# Call after the data is ready (after load_and_prepare())
visualize_grade(marvin_data, "marvin")
visualize_grade(mclaughlin_data, "mclaughlin")
"""
