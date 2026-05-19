# gcn.py
# A Graph Convolutional Network that predicts ore grade at each block node.
#
# WHY A GNN HERE?
# A mine block model is a graph: each block has spatial neighbours, and
# the grade of a block is geologically influenced by surrounding blocks.
# A GNN formalizes this by passing messages between neighbours —
# each block aggregates its neighbours' features to build a richer representation,
# then predicts grade from that enriched representation.
#
# MESSAGE PASSING (what GCNConv actually does):
#   For each node i, collect features from all neighbours j connected by an edge.
#   Aggregate them (sum/mean), combine with node i's own features,
#   apply a linear transformation + normalization. That's one GCNConv layer.
#   Stacking two layers means each node can "see" two hops away in the graph.

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GradeGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64, out_channels: int = 1):
        """
        Args:
            in_channels:  number of input node features
                          this must match data.x.shape[1] — either 4 (McLaughlin) or 5 (Marvin)
                          OR the embedding_dim if using CNN encoder output as input
            hidden_dim:   size of the internal representation after the first GCN layer
            out_channels: 1 for single grade prediction (AU only)
        """
        super().__init__()

        # Input projection layer — this is the fix for the feature dimension mismatch.
        # Marvin has 5 features, McLaughlin has 4, but the GCN hidden layers
        # need a consistent dimension. This linear layer maps any input size → hidden_dim,
        # so the same GCN architecture works on both mines without modification.
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Two GCNConv layers.
        # Layer 1: each node aggregates its direct neighbours (1 hop)
        # Layer 2: each node aggregates neighbours-of-neighbours (2 hops)
        # Two hops is enough for spatial grade estimation — beyond that,
        # the signal gets diluted by distant unrelated blocks.
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Output head: a linear layer maps the final node representation to a grade prediction.
        # out_channels=1 means one grade value per block.
        self.output_head = nn.Linear(hidden_dim, out_channels)

        self.relu = nn.ReLU()

        # Dropout for regularization — randomly zeroes 20% of neuron activations during training.
        # Prevents the model from memorizing specific blocks rather than learning
        # generalizable geological patterns. Disabled automatically during model.eval().
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          node feature matrix, shape [num_nodes, in_channels]
            edge_index: graph connectivity, shape [2, num_edges]

        Returns:
            grade predictions, shape [num_nodes, 1]
        """
        # print(f"GNN input — x: {x.shape}, edge_index: {edge_index.shape}")

        # Project input features to hidden_dim
        x = self.relu(self.input_proj(x))
        # shape: [num_nodes, hidden_dim]

        # First message passing layer — each block sees its 8 direct neighbours
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        # shape: [num_nodes, hidden_dim]

        # Second message passing layer — each block now sees 2 hops away
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        # shape: [num_nodes, hidden_dim]

        # Predict grade from the enriched node representation
        out = self.output_head(x)
        # shape: [num_nodes, 1]

        # print(f"GNN output shape: {out.shape}")
        return out