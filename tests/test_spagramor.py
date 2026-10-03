import unittest
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml
from scipy import sparse

from src.clustering.modality_aware_readout import modality_aware_readout
from experiments.run_exp import (
    _effective_coefficients,
    _load_checkpoint,
    _save_checkpoint,
)
from src.models.multigranularity import (
    CoarseOnlyFusion,
    FineOnlyFusion,
    LinearFusion,
    MultigranularityFusion,
    ViewMultiGranularityEncoder,
)
from src.models.spagramor import SpaGraMOR
from scripts.validate_formal_protocol import (
    validate_ablation_config,
    validate_formal_config,
)


class SpaGraMORContractTest(unittest.TestCase):
    @staticmethod
    def _readout_config(mode="modality_aware"):
        return {
            "mode": mode,
            "bsrr": {"enabled": True, "spatial_k": 3},
            "adt": {"method": "kmeans", "n_init": 20, "random_state": 0},
            "atac": {
                "method": "spectral",
                "latent_k": 10,
                "assign_labels": "kmeans",
                "n_init": 20,
                "random_state": 0,
                "eigen_solver": "arpack",
            },
        }

    def _embedding_and_coordinates(self):
        rng = np.random.default_rng(4)
        return (
            rng.normal(size=(30, 8)).astype(np.float32),
            rng.normal(size=(30, 2)).astype(np.float32),
        )

    def test_adt_routes_to_kmeans(self):
        embedding, coordinates = self._embedding_and_coordinates()
        _, predicted, diagnostics = modality_aware_readout(
            embedding,
            coordinates,
            second_modality="ADT",
            n_clusters=4,
            readout_config=self._readout_config(),
        )
        self.assertEqual(predicted.shape, (30,))
        self.assertEqual(diagnostics["downstream"]["method"], "kmeans")
        self.assertEqual(diagnostics["downstream"]["n_init"], 20)
        self.assertEqual(diagnostics["downstream"]["random_state"], 0)

    def test_fusion_modes_preserve_shapes(self):
        x = torch.randn(12, 32)
        for mode in ("full", "fine_only", "coarse_only", "linear_fusion"):
            encoder = ViewMultiGranularityEncoder(
                input_dim=32,
                fine_dim=128,
                coarse_dim=16,
                output_dim=16,
                fusion_mode=mode,
            )
            result = encoder(x)
            self.assertEqual(tuple(result["z"].shape), (12, 128))
            self.assertEqual(tuple(result["h"].shape), (12, 16))
            self.assertEqual(tuple(result["g"].shape), (12, 16))
            self.assertTrue(torch.isfinite(result["g"]).all())

    def test_atac_readout_is_reproducible(self):
        embedding, coordinates = self._embedding_and_coordinates()
        config = self._readout_config()
        first = modality_aware_readout(
            embedding, coordinates, second_modality="ATAC", n_clusters=4, readout_config=config
        )
        second = modality_aware_readout(
            embedding, coordinates, second_modality="ATAC", n_clusters=4, readout_config=config
        )
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2]["downstream"]["method"], "spectral")
        self.assertEqual(first[2]["downstream"]["affinity_dtype"], "float64")
        self.assertEqual(first[2]["downstream"]["latent_k"], 10)

    def test_kmeans_all_control_uses_kmeans_for_atac(self):
        embedding, coordinates = self._embedding_and_coordinates()
        _, _, diagnostics = modality_aware_readout(
            embedding,
            coordinates,
            second_modality="ATAC",
            n_clusters=4,
            readout_config=self._readout_config("kmeans_all"),
        )
        self.assertEqual(diagnostics["downstream"]["method"], "kmeans")

    def test_unsupported_modality_raises(self):
        embedding, coordinates = self._embedding_and_coordinates()
        with self.assertRaises(ValueError):
            modality_aware_readout(
                embedding,
                coordinates,
                second_modality="RNA",
                n_clusters=4,
                readout_config=self._readout_config(),
            )

    def test_disabled_bsrr_preserves_concat_z(self):
        embedding, coordinates = self._embedding_and_coordinates()
        config = self._readout_config()
        config["bsrr"]["enabled"] = False
        z_final, _, diagnostics = modality_aware_readout(
            embedding,
            coordinates,
            second_modality="ADT",
            n_clusters=4,
            readout_config=config,
        )
        self.assertTrue(np.array_equal(z_final, embedding))
        self.assertFalse(diagnostics["refinement"]["enabled"])

    def test_fusion_ablation_semantics(self):
        z = torch.randn(8, 128)
        h = torch.randn(8, 16)
        z_other = torch.randn_like(z)
        h_other = torch.randn_like(h)
        fine = FineOnlyFusion(128, 16)
        coarse = CoarseOnlyFusion(16, 16)
        linear = LinearFusion(128, 16, 16)
        full = MultigranularityFusion(128, 16, 16)
        self.assertTrue(torch.allclose(fine(z, h), fine(z, h_other)))
        self.assertTrue(torch.allclose(coarse(z, h), coarse(z_other, h)))
        self.assertFalse(torch.allclose(linear(z, h), linear(z_other, h_other)))
        self.assertFalse(torch.allclose(full(z, h), full(z_other, h_other)))

    def test_frozen_formal_and_ablation_protocol(self):
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "configs" / "formal").glob("*.yaml")):
            validate_formal_config(yaml.safe_load(path.read_text(encoding="utf-8")), path)
        ablation_path = root / "configs" / "ablation" / "base_ablation.yaml"
        validate_ablation_config(yaml.safe_load(ablation_path.read_text(encoding="utf-8")), ablation_path)

    def test_cluster_warmup_schedule(self):
        configured = (1.0, 3.0, 0.1)
        ablation = {"reconstruction": True, "sample_contrastive": True, "cluster_contrastive": True}
        for epoch, expected in ((1, 0.0), (10, 0.0), (11, 0.1)):
            _, _, cluster, _ = _effective_coefficients(
                configured, epoch=epoch, warm_up_epochs=10, ablation=ablation
            )
            self.assertEqual(cluster, expected)

    def test_uniform_view_weights_are_exactly_equal(self):
        torch.manual_seed(7)
        model = SpaGraMOR(
            {"RNA": 4, "ADT": 4},
            gcn_hidden_dim=8,
            fine_dim=6,
            coarse_dim=3,
            representation_dim=3,
            num_clusters=3,
            fusion_hidden_dim=5,
            view_weighting="uniform",
        )
        n_spots = 12
        x_rna = torch.randn(n_spots, 4)
        x_adt = torch.randn(n_spots, 4)
        adjacency = sparse.eye(n_spots, format="csr")
        with torch.no_grad():
            output = model(
                x_rna, adjacency, adjacency,
                x_adt, adjacency, adjacency,
                lambda_rec=1.0,
                lambda_mgcl=3.0,
                lambda_cluster=0.0,
            )
        expected = torch.full((4,), 0.25, dtype=output.weights.dtype)
        self.assertTrue(torch.equal(output.weights.cpu(), expected))
        self.assertEqual(model.cluster_head.projection.in_features, 6)

    def test_checkpoint_restores_optimizer_history_and_rng(self):
        torch.manual_seed(12)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss = model(torch.ones(4, 3)).sum()
        loss.backward()
        optimizer.step()
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            history = [{"epoch": 3, "effective_lambda_cluster": 0.1}]
            _save_checkpoint(checkpoint, model, optimizer, 3, history)
            expected_python = random.random()
            expected_numpy = np.random.random()
            expected_torch = torch.rand(1)

            restored_model = torch.nn.Linear(3, 2)
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.01)
            completed, restored_history = _load_checkpoint(
                checkpoint, restored_model, restored_optimizer, torch.device("cpu")
            )
            self.assertEqual(completed, 3)
            self.assertEqual(restored_history, history)
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.random(), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(1), expected_torch))
            for left, right in zip(model.parameters(), restored_model.parameters()):
                self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
