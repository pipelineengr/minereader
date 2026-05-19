# schemas.py
# Pydantic models define the shape of data coming INTO and going OUT of the API.
# Pydantic validates automatically — if a request sends a string where a float
# is expected, it rejects it before the data ever touches the model.
# Think of schemas as contracts: "this is exactly what I accept and return."

from pydantic import BaseModel, field_validator, model_validator
from typing import Optional
import math

class BlockNode(BaseModel):
    """
    A single mine block with spatial coordinates and optional assay features.
    Each BlockNode becomes one node in the PyG graph inside the endpoint.
    """
    x: float                        # easting coordinate
    y: float                        # northing coordinate
    z: float                        # elevation coordinate
    density: Optional[float] = None # rock density — optional, defaults to mean if missing
    copper: Optional[float] = None  # copper grade — optional, McLaughlin has none

    @field_validator("x", "y", "z")
    @classmethod
    def coords_must_be_finite(cls, v):
        # Reject NaN or infinite coordinates — these would silently corrupt the graph
        if not math.isfinite(v):
            raise ValueError(f"Coordinate must be finite, got {v}")
        return v


class EdgePair(BaseModel):
    """
    A directed edge between two block nodes.
    source and target are indices into the MineGraph.nodes list.
    """
    source: int   # index of the source node in the nodes list
    target: int   # index of the target node in the nodes list


class MineGraph(BaseModel):
    """
    The full graph submitted to /estimate-blocks.
    Contains a list of blocks (nodes) and their spatial connections (edges).
    Pydantic validates the entire structure before any model inference runs.
    """
    mine_id: str                    # identifier string, e.g. "marvin" or "mclaughlin"
    nodes: list[BlockNode]          # list of mine blocks
    edges: list[EdgePair]           # spatial adjacency pairs

    @model_validator(mode="after")
    def validate_graph(self):
        n = len(self.nodes)

        # Minimum node count — a graph with fewer than 10 blocks
        # can't produce meaningful spatial grade predictions
        if n < 10:
            raise ValueError(f"Graph must have at least 10 nodes, got {n}")

        # Validate all edge indices are within bounds
        for i, edge in enumerate(self.edges):
            if edge.source >= n or edge.target >= n:
                raise ValueError(
                    f"Edge {i} references node index out of range: "
                    f"source={edge.source}, target={edge.target}, num_nodes={n}"
                )
            if edge.source == edge.target:
                raise ValueError(f"Edge {i} is a self-loop (source == target == {edge.source})")

        # Check for isolated nodes — a node with no edges can't participate
        # in message passing and will always output the same value
        connected = set()
        for edge in self.edges:
            connected.add(edge.source)
            connected.add(edge.target)
        isolated = [i for i in range(n) if i not in connected]
        if isolated:
            raise ValueError(f"Graph has {len(isolated)} isolated nodes: {isolated[:5]}...")

        return self


class PredictionResult(BaseModel):
    """
    Response from /estimate-blocks.
    Contains per-block grade predictions and aggregate metrics.
    """
    mine_id: str
    num_blocks: int
    predictions: list[float]        # predicted grade per block, real units (g/t)
    mean_predicted_grade: float     # average predicted grade across all blocks
    model_version: str = "best"     # which checkpoint was used


class AdaptRequest(BaseModel):
    """
    Request body for /adapt-model.
    Provides labeled blocks from a new mine for few-shot fine-tuning.
    """
    mine_id: str
    graph: MineGraph
    labels: list[float]             # known grade values for each node
    epochs: int = 20                # how many fine-tuning steps to run
    learning_rate: float = 1e-4     # small LR preserves pre-trained features

    @field_validator("labels")
    @classmethod
    def labels_must_be_positive(cls, v):
        if any(g < 0 for g in v):
            raise ValueError("Grade labels must be non-negative")
        return v


class AdaptResponse(BaseModel):
    """Response from /adapt-model."""
    mine_id: str
    epochs_run: int
    final_mae: float                # MAE on the provided labeled data after fine-tuning


class DriftSnapshot(BaseModel):
    """A single model evaluation snapshot from one mine visit."""
    visit_number: int
    mae: float                      # Mean Absolute Error at this visit
    r2: float                       # R-squared at this visit

    @field_validator("mae")
    @classmethod
    def mae_non_negative(cls, v):
        if v < 0:
            raise ValueError("MAE cannot be negative")
        return v

    @field_validator("r2")
    @classmethod
    def r2_in_range(cls, v):
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f"R² must be between -1 and 1, got {v}")
        return v


class TrackPerfRequest(BaseModel):
    """Request body for /track-perf."""
    mine_id: str
    snapshots: list[DriftSnapshot]  # ordered list of evaluation snapshots

    @field_validator("snapshots")
    @classmethod
    def min_snapshots(cls, v):
        # Need at least 3 visits to detect a trend — fewer is meaningless
        if len(v) < 3:
            raise ValueError(f"Need at least 3 snapshots to detect drift, got {len(v)}")
        return v


class TrackPerfResponse(BaseModel):
    """Response from /track-perf."""
    mine_id: str
    drift_risk_score: float         # 0.0 to 1.0 — probability of drift
    recommendation: str             # plain-English action for the geologist
    num_snapshots: int