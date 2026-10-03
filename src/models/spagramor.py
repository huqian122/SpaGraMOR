"""SpaGraMOR model with explicit training ablation switches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import torch
from torch import Tensor, nn

from src.losses.cluster_contrastive import ClusterContrastiveLoss, ClusterHead
from src.losses.reconstruction import ReconstructionLoss
from src.losses.sample_contrastive import AdaptiveSampleContrastiveLoss
from src.models.adaptive_weight import AdaptiveWeightModule
from src.models.gcn import GraphViewGCN
from src.models.multigranularity import ViewMultiGranularityEncoder


@dataclass
class SpaGraMORForwardOutput:
    total_loss: Tensor
    reconstruction_loss: Tensor
    sample_contrastive_loss: Tensor
    cluster_contrastive_loss: Tensor
    global_representation: Tensor
    wd_distances: Tensor
    weights: Tensor
    mgcl_weight_std: Tensor
    neg_count: int
    snf_masked_positions: int
    weighted_representation: Tensor
    mean_representation: Tensor
    cluster_assignments: Sequence[Tensor]
    gcn_views: Dict[str, Tensor]
    multigranularity_views: Dict[str, Dict[str, Tensor]]


class SpaGraMOR(nn.Module):
    """Four-view MGCMVC backbone with config-controlled ablations.

    The cluster head always consumes fine representations ``Z_v``. The
    downstream readout is intentionally outside this module.
    """

    def __init__(
        self,
        input_dims: Mapping[str, int],
        gcn_hidden_dim: int = 32,
        fine_dim: int = 128,
        coarse_dim: int = 16,
        representation_dim: int = 16,
        num_clusters: int = 7,
        fusion_hidden_dim: int = 64,
        temperature: float = 0.3,
        cluster_temperature: float = 0.3,
        cluster_regularization_weight: float = 1.0,
        multigranularity_mode: str = "full",
        view_weighting: str = "adaptive",
    ) -> None:
        super().__init__()
        if len(input_dims) != 2:
            raise ValueError("SpaGraMOR expects exactly two modalities")
        if view_weighting not in {"adaptive", "uniform"}:
            raise ValueError("view_weighting must be adaptive or uniform")
        self.modality_names = tuple(input_dims)
        self.view_weighting = view_weighting
        self.gcn = GraphViewGCN(input_dims, hidden_dim=gcn_hidden_dim)
        first_modality, second_modality = self.modality_names
        self.view_order = (
            f"{first_modality.lower()}_spatial",
            f"{first_modality.lower()}_feature",
            f"{second_modality.lower()}_spatial",
            f"{second_modality.lower()}_feature",
        )
        self.view_encoders = nn.ModuleDict(
            {
                view_name: ViewMultiGranularityEncoder(
                    input_dim=gcn_hidden_dim,
                    fine_dim=fine_dim,
                    coarse_dim=coarse_dim,
                    output_dim=representation_dim,
                    fusion_mode=multigranularity_mode,
                )
                for view_name in self.view_order
            }
        )
        self.weight_module = AdaptiveWeightModule(
            num_views=len(self.view_order),
            representation_dim=representation_dim,
            global_dim=representation_dim,
            fusion_hidden_dim=fusion_hidden_dim,
        )
        self.cluster_head = ClusterHead(fine_dim, num_clusters)
        self.reconstruction_loss = ReconstructionLoss()
        self.sample_contrastive_loss = AdaptiveSampleContrastiveLoss(
            temperature=temperature
        )
        self.cluster_contrastive_loss = ClusterContrastiveLoss(
            temperature=cluster_temperature
        )
        self.cluster_regularization_weight = cluster_regularization_weight

    def forward(
        self,
        modality_a_features: object,
        modality_a_spatial_adj: object,
        modality_a_feature_adj: object,
        modality_b_features: object,
        modality_b_spatial_adj: object,
        modality_b_feature_adj: object,
        lambda_rec: float = 1.0,
        lambda_mgcl: float = 1.0,
        lambda_cluster: float = 1.0,
        modality_a_name: str | None = None,
        modality_b_name: str | None = None,
    ) -> SpaGraMORForwardOutput:
        coefficients = (lambda_rec, lambda_mgcl, lambda_cluster)
        if any(float(value) < 0 for value in coefficients):
            raise ValueError("loss coefficients must be nonnegative")
        modality_a_name = modality_a_name or self.modality_names[0]
        modality_b_name = modality_b_name or self.modality_names[1]
        gcn_views = self.gcn(
            modality_a_features,
            modality_a_spatial_adj,
            modality_a_feature_adj,
            modality_b_features,
            modality_b_spatial_adj,
            modality_b_feature_adj,
            modality_a_name=modality_a_name,
            modality_b_name=modality_b_name,
        )
        multigranularity_views = {
            view_name: self.view_encoders[view_name](gcn_views[view_name])
            for view_name in self.view_order
        }
        representations = [
            multigranularity_views[view_name]["g"] for view_name in self.view_order
        ]
        reconstructions = [
            multigranularity_views[view_name]["reconstruction"]
            for view_name in self.view_order
        ]
        global_representation, wd_distances, adaptive_weights = self.weight_module(
            representations,
        )
        if self.view_weighting == "uniform":
            weights = torch.full_like(adaptive_weights, 1.0 / len(representations))
        else:
            weights = adaptive_weights

        reconstruction = self.reconstruction_loss(
            [gcn_views[view_name] for view_name in self.view_order], reconstructions
        )
        sample_contrastive, sample_debug = self.sample_contrastive_loss(
            representations, weights, return_debug=True
        )
        assignments = [
            self.cluster_head(multigranularity_views[view_name]["z"])
            for view_name in self.view_order
        ]
        cluster_contrastive = self.cluster_contrastive_loss(
            assignments, regularization_weight=self.cluster_regularization_weight
        )
        stacked = torch.stack(representations, dim=0)
        weighted_representation = torch.sum(weights[:, None, None] * stacked, dim=0)
        mean_representation = stacked.mean(dim=0)
        total = (
            lambda_rec * reconstruction
            + lambda_mgcl * sample_contrastive
            + lambda_cluster * cluster_contrastive
        )
        return SpaGraMORForwardOutput(
            total_loss=total,
            reconstruction_loss=reconstruction,
            sample_contrastive_loss=sample_contrastive,
            cluster_contrastive_loss=cluster_contrastive,
            global_representation=global_representation,
            wd_distances=wd_distances,
            weights=weights,
            mgcl_weight_std=weights.detach().std(unbiased=False),
            neg_count=int(sample_debug["neg_count"]),
            snf_masked_positions=int(sample_debug["masked_positions"]),
            weighted_representation=weighted_representation,
            mean_representation=mean_representation,
            cluster_assignments=assignments,
            gcn_views=gcn_views,
            multigranularity_views=multigranularity_views,
        )
