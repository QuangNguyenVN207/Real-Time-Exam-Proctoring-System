from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.ai_services.face_verify.face_verify import FaceVerifier
from backend.core.config import settings


def verifier_with_gallery(names: list[str], vectors: np.ndarray) -> FaceVerifier:
    verifier = FaceVerifier.__new__(FaceVerifier)
    verifier._faiss_index = None
    verifier._set_database(names, vectors)
    verifier.similarity_threshold = 0.32
    verifier.identity_margin_threshold = 0.08
    return verifier


class FaceVerifierLogicTests(unittest.TestCase):
    def test_locked_release_config_matches_runtime(self) -> None:
        release_path = Path(__file__).with_name("FACE_P1_RELEASE.json")
        runtime = json.loads(release_path.read_text(encoding="utf-8"))["runtime"]
        self.assertEqual(runtime["model_pack"], settings.face_model_name)
        self.assertEqual(runtime["detector_input_size"], list(settings.face_det_size))
        self.assertEqual(
            runtime["detector_confidence_threshold"],
            settings.face_detection_threshold,
        )
        self.assertEqual(runtime["similarity_threshold"], settings.face_similarity_threshold)
        self.assertEqual(
            runtime["runner_up_margin_threshold"],
            settings.face_identity_margin_threshold,
        )

    def test_best_two_matches_normalizes_vectors(self) -> None:
        verifier = verifier_with_gallery(
            ["P01", "P02"],
            np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        )
        index, top1, top2 = verifier._best_two_matches(
            np.asarray([4.0, 0.0], dtype=np.float32)
        )
        self.assertEqual(index, 0)
        self.assertAlmostEqual(top1, 1.0)
        self.assertAlmostEqual(top2, 0.0)

    def test_margin_rejects_ambiguous_identity(self) -> None:
        verifier = verifier_with_gallery(
            ["P01", "P02"],
            np.asarray([[1.0, 0.0], [0.99, 0.10]], dtype=np.float32),
        )
        _, top1, top2 = verifier._best_two_matches(
            np.asarray([1.0, 0.05], dtype=np.float32)
        )
        self.assertGreater(top1, verifier.similarity_threshold)
        self.assertLess(top1 - top2, verifier.identity_margin_threshold)

    def test_identify_rejects_ambiguous_top_two(self) -> None:
        verifier = verifier_with_gallery(
            ["P01", "P02"],
            np.asarray([[1.0, 0.0], [0.99, 0.10]], dtype=np.float32),
        )
        verifier._detect_faces = lambda _region: [
            SimpleNamespace(
                bbox=np.asarray([0, 0, 5, 5]),
                normed_embedding=np.asarray([1.0, 0.05], dtype=np.float32),
            )
        ]
        self.assertIsNone(verifier.identify(np.zeros((10, 10, 3), dtype=np.uint8)))

    def test_assigned_identity_requires_expected_actor_to_be_top_one(self) -> None:
        verifier = verifier_with_gallery(
            ["P01", "P02"],
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        verifier._detect_faces = lambda _region: [
            SimpleNamespace(
                bbox=np.asarray([0, 0, 5, 5]),
                normed_embedding=np.asarray([0.0, 1.0], dtype=np.float32),
            )
        ]
        alert = verifier.verify_assigned_identity(
            np.zeros((10, 10, 3), dtype=np.uint8),
            [0, 0, 10, 10],
            "P01",
            1.25,
        )
        self.assertIsNotNone(alert)
        self.assertEqual(alert["details"]["matched_student_id"], "P02")
        self.assertEqual(alert["details"]["decision_reason"], "wrong_top1_identity")

    def test_load_gallery_uses_actor_centroids(self) -> None:
        verifier = FaceVerifier.__new__(FaceVerifier)
        verifier._faiss_index = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gallery_embeddings.npz"
            np.savez_compressed(
                path,
                actor_ids=np.asarray(["P01", "P02"]),
                centroids=np.asarray([[2.0, 0.0] + [0.0] * 510, [0.0, 3.0] + [0.0] * 510], dtype=np.float32),
            )
            verifier._load_gallery(str(path))
        self.assertEqual(verifier._known_names, ["P01", "P02"])
        np.testing.assert_allclose(
            np.linalg.norm(verifier._known_vectors, axis=1),
            np.ones(2),
        )


if __name__ == "__main__":
    unittest.main()
