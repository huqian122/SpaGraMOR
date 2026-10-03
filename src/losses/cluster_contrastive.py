"""MGCMVC-style cluster-level contrastive loss."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ClusterHead(nn.Module):
    """Map each view's clustering representation to soft assignments ``Q_v``.

    In MGCMVC the clustering representation is the fine-grained encoder
    output ``Z_v``; the caller is responsible for passing that representation.
    """

    def __init__(self, input_dim: int, num_clusters: int) -> None:
        super().__init__()
        if num_clusters < 2:
            raise ValueError("num_clusters must be at least 2")
        self.projection = nn.Linear(input_dim, num_clusters)

    def forward(self, representation: Tensor) -> Tensor:
        if representation.ndim != 2:
            raise ValueError("representation must be 2D")
        return F.softmax(self.projection(representation), dim=1)

    @torch.no_grad()
    def initialize_from_kmeans(
        self,
        representations: Sequence[Tensor],
        *,
        seed: int = 0,
        temperature: float = 1.0,
    ) -> None:
        """Initialize logits from unsupervised warm-up clusters.

        The cluster loss is permutation-invariant but has a nearly uniform
        stationary point. Initializing the shared head from warm-up features
        gives its first active epoch a useful partition without using labels.
        For normalized representations, centers approximate cosine prototypes.
        """

        if not representations:
            raise ValueError("at least one representation is required")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        embedding = torch.stack(list(representations), dim=0).mean(dim=0)
        if embedding.ndim != 2 or embedding.shape[1] != self.projection.in_features:
            raise ValueError("representations do not match ClusterHead input dimension")

        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=self.projection.out_features,
            n_init=20,
            random_state=seed,
        )
        kmeans.fit(embedding.detach().cpu().numpy())
        centers = torch.as_tensor(
            kmeans.cluster_centers_,
            dtype=self.projection.weight.dtype,
            device=self.projection.weight.device,
        )
        self.projection.weight.copy_(centers / temperature)
        self.projection.bias.copy_(-centers.pow(2).sum(dim=1) / (2.0 * temperature))


def target_distribution(q: Tensor, eps: float = 1e-12) -> Tensor:
    """Construct MGCMVC's sharpened target distribution ``P``."""

    if q.ndim != 2:
        raise ValueError("q must be a 2D probability matrix")
    if torch.any(q < 0):
        raise ValueError("q must contain nonnegative probabilities")
    weight = q.pow(2) / q.sum(dim=0, keepdim=True).clamp_min(eps)
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(eps)


class ClusterContrastiveLoss(nn.Module):
    """MGCMVC cluster-level NT-Xent over corresponding view clusters.

    For each pair of views, the C cluster columns from both target matrices
    form 2C samples. A cluster is positive only with its same-index cluster
    in the other view; all remaining clusters are negatives.
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-12) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.eps = eps

    def pair_loss(self, q_left: Tensor, q_right: Tensor) -> Tensor:
        if q_left.shape != q_right.shape or q_left.ndim != 2:
            raise ValueError("q_left and q_right must have equal 2D shapes")
        p_left = target_distribution(q_left, self.eps)
        p_right = target_distribution(q_right, self.eps)
        # Cluster vectors are columns of P, matching MGCMVC's implementation.
        clusters = F.normalize(torch.cat((p_left.T, p_right.T), dim=0), dim=1)
        similarity = clusters @ clusters.T / self.temperature
        count = similarity.shape[0]
        num_clusters = q_left.shape[1]
        positive_index = (torch.arange(count, device=similarity.device) + num_clusters) % count
        self_mask = torch.eye(count, device=similarity.device, dtype=torch.bool)
        positive_mask = F.one_hot(positive_index, num_classes=count).to(torch.bool)
        negative_mask = ~(self_mask | positive_mask)
        positive_logits = similarity.gather(1, positive_index.unsqueeze(1))
        negative_logits = similarity.masked_fill(~negative_mask, float("-inf"))
        logits = torch.cat((positive_logits, negative_logits), dim=1)
        labels = torch.zeros(count, device=similarity.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)

    def entropy_regularizer(self, q: Tensor) -> Tensor:
        cluster_mass = target_distribution(q, self.eps).mean(dim=0)
        return torch.log(torch.tensor(q.shape[1], device=q.device, dtype=q.dtype)) + (
            cluster_mass * torch.log(cluster_mass.clamp_min(self.eps))
        ).sum()

    def diagnostics(self, assignments: Sequence[Tensor]) -> Dict[str, Tensor]:
        """Return non-gradient collapse indicators for experiment logging."""

        with torch.no_grad():
            q = torch.stack([assignment.detach() for assignment in assignments]).mean(dim=0)
            mass = q.mean(dim=0)
            sample_entropy = -(q * torch.log(q.clamp_min(self.eps))).sum(dim=1).mean()
            max_probability = q.max(dim=1).values.mean()
            effective_clusters = torch.exp(
                -(mass * torch.log(mass.clamp_min(self.eps))).sum()
            )
            return {
                "assignment_entropy": sample_entropy,
                "max_probability": max_probability,
                "effective_clusters": effective_clusters,
                "min_cluster_mass": mass.min(),
                "max_cluster_mass": mass.max(),
            }

    def forward(
        self,
        assignments: Sequence[Tensor],
        regularization_weight: float = 1.0,
    ) -> Tensor:
        if len(assignments) < 2:
            raise ValueError("at least two view assignments are required")
        total = assignments[0].new_zeros(())
        pair_count = 0
        for left_index in range(len(assignments)):
            for right_index in range(left_index + 1, len(assignments)):
                total = total + self.pair_loss(
                    assignments[left_index], assignments[right_index]
                )
                pair_count += 1
        contrastive = total / pair_count
        regularizer = torch.stack(
            [self.entropy_regularizer(assignment) for assignment in assignments]
        ).mean()
        return contrastive + regularization_weight * regularizer


def cluster_contrastive_loss(
    assignments: Sequence[Tensor],
    temperature: float = 1.0,
    regularization_weight: float = 1.0,
) -> Tensor:
    return ClusterContrastiveLoss(
        temperature=temperature,
    )(assignments, regularization_weight=regularization_weight)
