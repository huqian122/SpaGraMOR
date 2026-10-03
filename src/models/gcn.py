"""Graph-specific GCN view generation for SpaGraMOR."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse


def _to_dense_tensor(x: object, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        tensor = x.float()
    elif hasattr(x, "toarray"):
        tensor = torch.from_numpy(np.asarray(x.toarray())).float()
    else:
        tensor = torch.from_numpy(np.asarray(x)).float()
    return tensor.to(device) if device is not None else tensor


def _to_sparse_tensor(adj: object, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(adj, torch.Tensor) and adj.is_sparse:
        tensor = adj.coalesce().float()
        return tensor.to(device) if device is not None else tensor
    if not sparse.issparse(adj):
        adj = sparse.coo_matrix(np.asarray(adj))
    adj = adj.tocoo().astype(np.float32)
    indices = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
    values = torch.tensor(adj.data, dtype=torch.float32)
    tensor = torch.sparse_coo_tensor(indices, values, size=adj.shape).coalesce()
    return tensor.to(device) if device is not None else tensor


def normalize_sparse_adjacency(adj: object, add_self_loops: bool = True) -> torch.Tensor:
    """Symmetric normalization for sparse adjacency tensors."""

    tensor = _to_sparse_tensor(adj)
    if add_self_loops:
        identity = torch.eye(tensor.size(0), device=tensor.device).to_sparse().coalesce()
        tensor = (tensor + identity).coalesce()
    rowsum = torch.sparse.sum(tensor, dim=1).to_dense()
    inv_sqrt = torch.pow(rowsum.clamp_min(1e-12), -0.5)
    row, col = tensor.indices()
    values = tensor.values() * inv_sqrt[row] * inv_sqrt[col]
    return torch.sparse_coo_tensor(tensor.indices(), values, tensor.size()).coalesce()


class GraphEncoder(nn.Module):
    """Single-layer graph encoder shared by spatial and feature views."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, features: object, adjacency: object) -> torch.Tensor:
        x = _to_dense_tensor(features)
        adj = normalize_sparse_adjacency(adjacency).to(x.device)
        x = self.dropout(x)
        x = self.linear(x)
        x = torch.sparse.mm(adj, x)
        return self.activation(x)


class GraphViewGCN(nn.Module):
    """Generate the four graph views used by SpaGraMOR.

    The same modality-specific encoder is reused for that modality's spatial
    and feature graphs so the two views stay comparable while feature graphs
    remain modality isolated.
    """

    def __init__(self, input_dims: Mapping[str, int], hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {modality: GraphEncoder(input_dim, hidden_dim, dropout=dropout) for modality, input_dim in input_dims.items()}
        )

    def forward(
        self,
        modality_a_features: object,
        modality_a_spatial_adj: object,
        modality_a_feature_adj: object,
        modality_b_features: object,
        modality_b_spatial_adj: object,
        modality_b_feature_adj: object,
        modality_a_name: str = "RNA",
        modality_b_name: str = "ADT",
    ) -> Dict[str, torch.Tensor]:
        if modality_a_name not in self.encoders:
            raise KeyError(f"Unknown modality: {modality_a_name}")
        if modality_b_name not in self.encoders:
            raise KeyError(f"Unknown modality: {modality_b_name}")

        a_encoder = self.encoders[modality_a_name]
        b_encoder = self.encoders[modality_b_name]

        outputs = {
            f"{modality_a_name.lower()}_spatial": a_encoder(modality_a_features, modality_a_spatial_adj),
            f"{modality_a_name.lower()}_feature": a_encoder(modality_a_features, modality_a_feature_adj),
            f"{modality_b_name.lower()}_spatial": b_encoder(modality_b_features, modality_b_spatial_adj),
            f"{modality_b_name.lower()}_feature": b_encoder(modality_b_features, modality_b_feature_adj),
        }
        return outputs
