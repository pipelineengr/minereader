# cli.py
# Command-line interface for MineReader.
# Provides a single entry point for all project operations —
# data processing, training, inference, and API server.
#
# Usage: minereader <command> [options]
# Run:   minereader --help    to see all commands

import argparse
import sys
import os

from minereader.config import PROCESSED_DATA_DIR

def cmd_prepare(args):
    """Process raw CSV files into PyG graph objects."""
    from minereader.data.graph import load_and_prepare

    datasets = args.datasets if args.datasets else ["marvin", "mclaughlin"]
    for name in datasets:
        csv_path = os.path.join("minereader", "data", "raw", f"{name}.csv")
        if not os.path.exists(csv_path):
            print(f"[ERROR] CSV not found: {csv_path}")
            print(f"        Place your CSV at {csv_path} and try again.")
            sys.exit(1)
        print(f"\nProcessing {name}...")
        load_and_prepare(name, csv_path)
    print("\n Data preparation complete.")


def cmd_train(args):
    """Train one or all models."""
    if args.model in ("grade", "all"):
        print("\n--- Training grade GNN on Marvin ---")
        from minereader.train.train_grade_mean import train
        train()

    if args.model in ("drift", "all"):
        print("\n--- Training drift transformer ---")
        from minereader.train.drift import train
        train()

    if args.model in ("transfer", "all"):
        print("\n--- Running transfer learning experiment on McLaughlin ---")
        from minereader.train.transfer import run_transfer_experiment
        run_transfer_experiment()

    print("\n Training complete. Checkpoints saved to runs/")


def cmd_predict(args):
    """
    Run grade prediction on a CSV of blocks.
    Expects columns: X, Y, Z (and optionally density, copper).
    Outputs predictions as a new CSV with a 'predicted_grade' column.
    """
    import torch
    import pandas as pd
    import numpy as np
    from torch_geometric.nn import knn_graph
    from torch_geometric.data import Data
    from minereader.models.gcn import GradeGNN
    from config import RUNS_DIR, KNN_K

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading blocks from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Found {len(df)} blocks with columns: {list(df.columns)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # Build feature matrix — use whatever columns are available
    coords = df[["X", "Y", "Z"]].values.astype("float32")
    from sklearn.preprocessing import StandardScaler
    coords_norm = StandardScaler().fit_transform(coords)

    features = [coords_norm]
    if "density" in df.columns:
        features.append(StandardScaler().fit_transform(df[["density"]].values.astype("float32")))
    if "copper" in df.columns or "B" in df.columns:
        col = "copper" if "copper" in df.columns else "B"
        features.append(StandardScaler().fit_transform(df[[col]].values.astype("float32")))

    x = torch.tensor(np.hstack(features), dtype=torch.float)
    pos = torch.tensor(coords_norm, dtype=torch.float)
    edge_index = knn_graph(pos, k=KNN_K, loop=False)

    data = Data(x=x, edge_index=edge_index).to(device)

    # Load model
    in_channels = x.shape[1]
    model = GradeGNN(in_channels=in_channels, hidden_dim=64).to(device)
    ckpt = os.path.join(RUNS_DIR, "marvin_grade", "best.pt")
    if not os.path.exists(ckpt):
        print(f"[ERROR] No trained model found at {ckpt}")
        print("        Run: minereader train --model grade")
        sys.exit(1)

    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    model.eval()

    with torch.no_grad():
        preds = model(data.x, data.edge_index).cpu().numpy().flatten()

    # Load grade scaling parameters for inverse transform
    processed_path = os.path.join("minereader", "data", "processed", "marvin.pt")
    if os.path.exists(processed_path):
        meta = torch.load(processed_path, weights_only=False, map_location="cpu")
        preds = preds * meta.grade_std + meta.grade_mean

    preds = np.clip(preds, 0, None)
    df["predicted_grade"] = preds

    output_path = args.output if args.output else args.input.replace(".csv", "_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"\n Predictions saved to {output_path}")
    print(f"   Mean predicted grade: {preds.mean():.4f} g/t")
    print(f"   Grade range: [{preds.min():.4f}, {preds.max():.4f}] g/t")


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn
    print(f"Starting MineReader API on http://localhost:{args.port}")
    print(f"Interactive docs: http://localhost:{args.port}/docs")
    uvicorn.run(
        "minereader.api.main:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
    )


def cmd_status(args):
    """Show what has been trained and what's ready to run."""
    import json
    from minereader.config import RUNS_DIR, PROCESSED_DATA_DIR

    print("\n=== MineReader Status ===\n")

    # Data
    print("── Processed Data ──")
    for name in ["marvin", "mclaughlin"]:
        path = os.path.join(PROCESSED_DATA_DIR, f"{name}.pt")
        if os.path.exists(path):
            import torch
            d = torch.load(path, weights_only=False, map_location="cpu")
            print(f"{name}: {d.num_nodes:,} nodes, {d.x.shape[1]} features")
        else:
            print(f"{name}: not processed — run: minereader prepare")

    # Models
    print("\n── Trained Models ──")
    checkpoints = {
        "Grade GNN (Marvin)":      os.path.join(RUNS_DIR, "marvin_grade", "best.pt"),
        "Drift Transformer":       os.path.join(RUNS_DIR, "drift_transformer", "best.pt"),
        "Fine-tuned (McLaughlin)": os.path.join(RUNS_DIR, "mclaughlin_finetune", "best.pt"),
        "Scratch (McLaughlin)":    os.path.join(RUNS_DIR, "mclaughlin_scratch", "best.pt"),
    }
    for name, path in checkpoints.items():
        status = "Yes" if os.path.exists(path) else ""
        print(f"  {status} {name}")

    # Results
    print("\n── Results ──")
    baseline_path = os.path.join(RUNS_DIR, "marvin_grade", "baseline_comparison.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            b = json.load(f)
        print(f"  Marvin — Mean predictor MAE : {b['mean_predictor_mae_real']:.4f} g/t")
        print(f"  Marvin — GNN best MAE       : {b['gnn_best_mae_real']:.4f} g/t")
        print(f"  Marvin — Improvement        : {b['improvement_pct']:.1f}%")
    else:
        print("   No baseline results — run: minereader train --model grade")

    transfer_path = os.path.join(RUNS_DIR, "transfer_experiment", "results.json")
    if os.path.exists(transfer_path):
        with open(transfer_path) as f:
            t = json.load(f)
        print(f"\n  McLaughlin — Zero-shot MAE  : {t['zero_shot_mae']:.4f} g/t")
        print(f"  McLaughlin — Fine-tuned MAE : {t['finetuned_mae']:.4f} g/t")
        print(f"  McLaughlin — Scratch MAE    : {t['scratch_mae']:.4f} g/t")
        print(f"  Pre-training value          : {t['gap_pretraining_value']:+.4f} g/t")
    else:
        print("   No transfer results — run: minereader train --model transfer")

    print()


def main():
    parser = argparse.ArgumentParser(
        prog="minereader",
        description="MineReader — Deep learning ore grade estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  prepare   Process raw CSV files into graph objects
  train     Train one or all models
  predict   Run grade prediction on a new CSV file
  serve     Start the FastAPI server
  status    Show what's trained and ready

examples:
  minereader prepare
  minereader prepare --datasets marvin
  minereader train --model all
  minereader train --model grade
  minereader predict --input data/raw/newmine.csv
  minereader serve
  minereader serve --port 8080 --no-reload
  minereader status
        """
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True

    # prepare
    p_prepare = subparsers.add_parser("prepare", help="Process raw CSVs into graph objects")
    p_prepare.add_argument(
        "--datasets", nargs="+", choices=["marvin", "mclaughlin"],
        help="Which datasets to process (default: both)"
    )

    # train
    p_train = subparsers.add_parser("train", help="Train models")
    p_train.add_argument(
        "--model", choices=["grade", "drift", "transfer", "all"],
        default="all",
        help="Which model to train (default: all)"
    )

    # predict
    p_predict = subparsers.add_parser("predict", help="Run grade prediction on a CSV")
    p_predict.add_argument("--input", required=True, help="Path to input CSV file")
    p_predict.add_argument("--output", help="Path to output CSV (default: input_predictions.csv)")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the FastAPI API server")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    p_serve.add_argument("--no-reload", dest="reload", action="store_false",
                         help="Disable auto-reload (use in production)")

    # status
    subparsers.add_parser("status", help="Show training status and results")

    args = parser.parse_args()

    dispatch = {
        "prepare":  cmd_prepare,
        "train":    cmd_train,
        "predict":  cmd_predict,
        "serve":    cmd_serve,
        "status":   cmd_status,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()