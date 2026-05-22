# MineReader

MineReader is a Python project that predicts ore grade from mine block data.

It turns block data into a graph, trains a graph neural network, and makes predictions through a small API.

## What it does

- Reads mine block CSV files.
- Builds a graph from nearby blocks.
- Trains a PyTorch model on the data.
- Predicts ore grade for new blocks.
- Serves predictions through FastAPI.

## Why it matters

Mining teams need a way to estimate where valuable material may be.
MineReader is to showcase how machine learning can help with that task.

## How it works

1. Load raw block data.
2. Clean and scale the data.
3. Build graph connections between blocks.
4. Train the model.
5. Save the trained model.
6. Use the model to make predictions.

## Tech stack (written in Python)

- PyTorch
- PyTorch Geometric
- FastAPI
- Pydantic
- Pandas
- NumPy

## Project layout

- `data/` - data prep code.
- `models/` - model code.
- `train/` - training scripts.
- `api/` -web API code.
- `cli.py` - command-line tool.

## Quick start

```bash
python cli.py prepare
python cli.py train --model grade
python cli.py serve
```

## Example

```bash
python cli.py predict --input sample.csv
```

## Notes

This project is for learning and portfolio use, so it'll be constantly changing.
It shows how graphs, machine learning, and APIs can work together.
