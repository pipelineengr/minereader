# main.py
# FastAPI application — four endpoints that expose the trained models as a web service.
#
# ENDPOINT OVERVIEW:
#   POST /estimate-blocks     — grade prediction on a new mine graph
#   POST /adapt-model         — few-shot fine-tuning on a new mine
#   POST /track-perf          — drift detection on deployment snapshots
#   GET  /geologist-report    — human-readable HTML report for a mine
#
# Run with: uvicorn minereader.api.main:app --reload

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from torch_geometric.data import Data
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from config import RUNS_DIR, PROCESSED_DATA_DIR
from models.gcn import GradeGNN
from models.transformer import DriftTransformer
from api.schemas import (
    MineGraph, PredictionResult,
    AdaptRequest, AdaptResponse,
    TrackPerfRequest, TrackPerfResponse,
)

# ------------------------------------------------------------------ #
# MODEL REGISTRY
# Models are loaded once at startup and reused for all requests.
# Loading a model on every request would add ~500ms latency each time.
# ------------------------------------------------------------------ #
models = {}

def load_models():
    """
    Load all trained model checkpoints into memory at application startup.
    Stored in the global `models` dict so endpoints can access them.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on: {device}")

    # --- Grade GNN ---
    # in_channels=5 for Marvin (x,y,z,density,copper)
    grade_model = GradeGNN(in_channels=5, hidden_dim=64).to(device)
    grade_ckpt = os.path.join(RUNS_DIR, "marvin_grade", "best.pt")
    if os.path.exists(grade_ckpt):
        grade_model.load_state_dict(
            torch.load(grade_ckpt, weights_only=True, map_location=device)
        )
        grade_model.eval()
        # eval() disables dropout so predictions are deterministic
        print(f"Loaded GradeGNN from {grade_ckpt}")
    else:
        print(f"WARNING: No GradeGNN checkpoint found at {grade_ckpt}")

    # --- Drift Transformer ---
    drift_model = DriftTransformer(input_dim=2, d_model=32, nhead=4, num_layers=2).to(device)
    drift_ckpt = os.path.join(RUNS_DIR, "drift_transformer", "best.pt")
    if os.path.exists(drift_ckpt):
        drift_model.load_state_dict(
            torch.load(drift_ckpt, weights_only=True, map_location=device)
        )
        drift_model.eval()
        print(f"Loaded DriftTransformer from {drift_ckpt}")
    else:
        print(f"WARNING: No DriftTransformer checkpoint found at {drift_ckpt}")

    # Load grade scaling parameters saved by prepare_graph.py
    # Needed to convert normalized predictions back to real g/t units
    grade_meta = {}
    for mine in ["marvin", "mclaughlin"]:
        meta_path = os.path.join(PROCESSED_DATA_DIR, f"{mine}.pt")
        if os.path.exists(meta_path):
            d = torch.load(meta_path, weights_only=False, map_location="cpu")
            grade_meta[mine] = {
                "grade_mean": d.grade_mean,
                "grade_std": d.grade_std,
            }

    models["grade"] = grade_model
    models["drift"] = drift_model
    models["device"] = device
    models["grade_meta"] = grade_meta
    print("All models loaded.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # lifespan runs load_models() once when the server starts,
    # then yields (server runs), then cleans up when server stops
    load_models()
    yield
    models.clear()

app = FastAPI(
    title="MineReader API",
    description="Deep learning ore grade estimation and model monitoring",
    version="0.1.0",
    lifespan=lifespan,
)


def minegraph_to_pyg(graph: MineGraph, device: torch.device) -> Data:
    """
    Convert a Pydantic MineGraph request into a PyG Data object.
    This is the serialization bridge between the JSON API layer and the model layer.

    JSON can't represent PyTorch tensors directly — this function does the conversion.
    Called inside every endpoint that needs to run the GNN.
    """
    # Build node feature matrix from BlockNode list
    features = []
    for node in graph.nodes:
        # Use 0.0 for missing optional features
        # In production you'd use the dataset mean, but 0.0 is fine for a portfolio project
        row = [
            node.x,
            node.y,
            node.z,
            node.density if node.density is not None else 0.0,
            node.copper if node.copper is not None else 0.0,
        ]
        features.append(row)

    x = torch.tensor(features, dtype=torch.float, device=device)
    # x shape: [num_nodes, 5]

    # Build edge_index from EdgePair list
    # edge_index must be shape [2, num_edges] in COO format
    if graph.edges:
        sources = [e.source for e in graph.edges]
        targets = [e.target for e in graph.edges]
        edge_index = torch.tensor(
            [sources, targets], dtype=torch.long, device=device
        )
    else:
        # Empty graph — no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    return Data(x=x, edge_index=edge_index, num_nodes=len(graph.nodes))


# ------------------------------------------------------------------ #
# ENDPOINT 1: POST /estimate-blocks
# Accepts a mine graph, returns per-block grade predictions.
# ------------------------------------------------------------------ #
@app.post("/estimate-blocks", response_model=PredictionResult)
async def estimate_blocks(graph: MineGraph):
    """
    Run grade estimation on a submitted mine block graph.
    Input: MineGraph with nodes (blocks) and edges (spatial connections)
    Output: PredictionResult with per-block grade estimates in g/t
    """
    if "grade" not in models:
        raise HTTPException(status_code=503, detail="Grade model not loaded")

    device = models["device"]
    model = models["grade"]

    # Convert Pydantic schema → PyG Data object
    data = minegraph_to_pyg(graph, device)

    # Normalize coordinates using the same StandardScaler logic as prepare_graph.py
    # In production this would use the saved joblib scaler — here we normalize inline
    pos = data.x[:, :3]
    pos_mean = pos.mean(dim=0)
    pos_std = pos.std(dim=0) + 1e-8  # add small epsilon to avoid division by zero
    data.x[:, :3] = (pos - pos_mean) / pos_std
    # 1e-8 is a tiny number added to prevent dividing by zero if all coords are identical

    # Run inference
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        # out shape: [num_nodes, 1] — normalized grade predictions

    # Convert normalized predictions back to real g/t units
    # Use Marvin scaling as default — in production you'd select by mine_id
    meta = models["grade_meta"].get(graph.mine_id, models["grade_meta"].get("marvin", {}))
    grade_std = meta.get("grade_std", 1.0)
    grade_mean = meta.get("grade_mean", 0.0)

    preds_real = (out.cpu().numpy().flatten() * grade_std) + grade_mean
    preds_real = np.clip(preds_real, 0, None).tolist()
    # Clip to 0 — negative grade predictions are physically impossible

    return PredictionResult(
        mine_id=graph.mine_id,
        num_blocks=len(graph.nodes),
        predictions=preds_real,
        mean_predicted_grade=float(np.mean(preds_real)),
    )


# ------------------------------------------------------------------ #
# ENDPOINT 2: POST /adapt-model
# Fine-tunes the grade model on labeled data from a new mine.
# ------------------------------------------------------------------ #
@app.post("/adapt-model", response_model=AdaptResponse)
async def adapt_model(request: AdaptRequest):
    """
    Fine-tune the grade model on a small set of labeled blocks from a new mine.
    This is the operational equivalent of the few-shot condition in train_transfer.py.
    """
    if "grade" not in models:
        raise HTTPException(status_code=503, detail="Grade model not loaded")

    if len(request.labels) != len(request.graph.nodes):
        raise HTTPException(
            status_code=422,
            detail=f"labels length ({len(request.labels)}) must match "
                   f"nodes length ({len(request.graph.nodes)})"
        )

    device = models["device"]

    # Build graph and label tensor
    data = minegraph_to_pyg(request.graph, device)
    y = torch.tensor(request.labels, dtype=torch.float, device=device).unsqueeze(1)
    # unsqueeze(1) reshapes [N] → [N, 1] to match model output shape

    # Fine-tune a copy of the loaded model
    # We copy weights so the base model in `models["grade"]` stays unchanged
    in_channels = data.x.shape[1]
    adapted_model = GradeGNN(in_channels=in_channels, hidden_dim=64).to(device)
    adapted_model.load_state_dict(models["grade"].state_dict())
    # load_state_dict copies all weights from the loaded model into the new instance

    optimizer = torch.optim.Adam(
        adapted_model.parameters(), lr=request.learning_rate
    )
    loss_fn = nn.MSELoss()

    adapted_model.train()
    for epoch in range(request.epochs):
        optimizer.zero_grad()
        out = adapted_model(data.x, data.edge_index)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

    # Compute final MAE on the provided labels
    adapted_model.eval()
    with torch.no_grad():
        final_out = adapted_model(data.x, data.edge_index)
        final_mae = torch.mean(torch.abs(final_out - y)).item()

    return AdaptResponse(
        mine_id=request.mine_id,
        epochs_run=request.epochs,
        final_mae=round(final_mae, 4),
    )


# ------------------------------------------------------------------ #
# ENDPOINT 3: POST /track-perf
# Detects model drift from a sequence of evaluation snapshots.
# ------------------------------------------------------------------ #
@app.post("/track-perf", response_model=TrackPerfResponse)
async def track_performance(request: TrackPerfRequest):
    """
    Run the drift transformer on a sequence of mine visit evaluation snapshots.
    Returns a drift risk score and plain-English recommendation.
    """
    if "drift" not in models:
        raise HTTPException(status_code=503, detail="Drift model not loaded")

    device = models["device"]
    model = models["drift"]

    # Build sequence tensor from snapshots
    # Sort by visit_number to ensure temporal order
    sorted_snaps = sorted(request.snapshots, key=lambda s: s.visit_number)
    seq = [[s.mae, s.r2] for s in sorted_snaps]
    # seq is a list of [MAE, R²] pairs in visit order

    x = torch.tensor([seq], dtype=torch.float, device=device)
    # Shape: [1, seq_len, 2] — batch of 1 sequence

    with torch.no_grad():
        drift_score = model(x).item()
    # drift_score: float between 0 and 1

    # Plain-English recommendation thresholds
    # These map a continuous probability to an actionable decision
    if drift_score > 0.75:
        recommendation = (
            "High drift risk detected. Model performance is degrading significantly. "
            "Schedule retraining before the next mine visit."
        )
    elif drift_score > 0.5:
        recommendation = (
            "Moderate drift risk. Monitor closely at next visit. "
            "Prepare labeled data for potential fine-tuning."
        )
    else:
        recommendation = (
            "Model performance is stable. No retraining required at this time."
        )

    return TrackPerfResponse(
        mine_id=request.mine_id,
        drift_risk_score=round(drift_score, 4),
        recommendation=recommendation,
        num_snapshots=len(request.snapshots),
    )


# ------------------------------------------------------------------ #
# ENDPOINT 4: GET /geologist-report/{mine_id}
# Returns an HTML page with grade heatmap and model metrics.
# ------------------------------------------------------------------ #
@app.get("/geologist-report/{mine_id}", response_class=HTMLResponse)
async def geologist_report(mine_id: str):
    """
    Generate a human-readable HTML report for a deployed mine model.
    Intended for geologists and business stakeholders who don't read JSON.
    Contains: grade heatmap, training metrics summary, transfer learning results.
    """
    # Load processed graph for visualization
    data_path = os.path.join(PROCESSED_DATA_DIR, f"{mine_id}.pt")
    if not os.path.exists(data_path):
        raise HTTPException(
            status_code=404,
            detail=f"No processed data found for mine_id '{mine_id}'. "
                   f"Run prepare_graph.py first."
        )

    data = torch.load(data_path, weights_only=False, map_location="cpu")

    # Subsample for visualization — 10k points renders smoothly in any browser
    VIZ_SAMPLE = 10000
    n = data.num_nodes
    if n > VIZ_SAMPLE:
        idx = torch.randperm(n)[:VIZ_SAMPLE]
        coords = data.pos[idx].numpy()
        grades = data.y[idx].numpy().flatten()
    else:
        coords = data.pos.numpy()
        grades = data.y.numpy().flatten()

    # Build Plotly 3D scatter as an HTML div (no separate file needed)
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=grades,
                colorscale="Viridis",
                colorbar=dict(title="Grade (norm.)"),
                opacity=0.7,
            ),
        )])
        fig.update_layout(
            title=f"{mine_id.title()} — Predicted Grade Distribution",
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        # to_html(full_html=False) returns just the <div> block, not a full page
        # We embed this div into our own HTML template below
        plot_div = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    except ImportError:
        plot_div = "<p>Plotly not installed — install with: pip install plotly</p>"

    # Load training metrics if available
    log_path = os.path.join(RUNS_DIR, f"{mine_id}_grade", "train_log.csv")
    metrics_html = ""
    if os.path.exists(log_path):
        import pandas as pd
        log = pd.read_csv(log_path)
        best_mae = log["val_mae_real"].min()
        best_epoch = log["val_mae_real"].idxmin() + 1
        metrics_html = f"""
        <div class="metrics">
            <h3>Model Performance</h3>
            <p><strong>Best Validation MAE:</strong> {best_mae:.4f} g/t
               (epoch {best_epoch} of {len(log)})</p>
        </div>
        """

    # Load transfer experiment results if available
    transfer_html = ""
    transfer_path = os.path.join(RUNS_DIR, "transfer_experiment", "results.json")
    if os.path.exists(transfer_path):
        with open(transfer_path) as f:
            tr = json.load(f)
        transfer_html = f"""
        <div class="metrics">
            <h3>Transfer Learning Results (McLaughlin)</h3>
            <table>
                <tr><th>Condition</th><th>MAE (g/t)</th></tr>
                <tr><td>Zero-shot (no McLaughlin data)</td>
                    <td>{tr['zero_shot_mae']}</td></tr>
                <tr><td>Fine-tuned (10% labels)</td>
                    <td>{tr['finetuned_mae']}</td></tr>
                <tr><td>Trained from scratch (10% labels)</td>
                    <td>{tr['scratch_mae']}</td></tr>
            </table>
            <p><strong>Pre-training value:</strong>
               {tr['gap_pretraining_value']:+.4f} g/t improvement over scratch</p>
        </div>
        """

    # Full HTML page — self-contained, no external dependencies except Plotly CDN
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MineReader — {mine_id.title()} Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            h1 {{ color: #2c3e50; }}
            h3 {{ color: #34495e; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
            .metrics {{ background: white; padding: 20px; border-radius: 8px;
                        margin: 20px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background: #2c3e50; color: white; }}
            tr:nth-child(even) {{ background: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>MineReader Geologist Report — {mine_id.title()}</h1>
        <p>Generated by MineReader v0.1.0 | Model: GCN + CNN Encoder | 
           Blocks visualized: {min(n, VIZ_SAMPLE):,} of {n:,}</p>
        {plot_div}
        {metrics_html}
        {transfer_html}
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# Health check — useful to confirm the server is running
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "cuda": torch.cuda.is_available(),
    }