"""Unit tests for Stage A1: Session manifest provenance."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pose_gaze.holistic.debug.session_manifest import (
    SessionManifest,
    SessionManifestRecorder,
    compute_model_artifacts_hashes,
    get_git_provenance,
    sha256_file,
    verify_model_artifacts,
)

from pose_gaze.settings import PROJECT_ROOT
REPO_ROOT = PROJECT_ROOT


class Sha256Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sha256_file_known_hash(self) -> None:
        f = self.tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        h = sha256_file(f)
        self.assertEqual(
            h,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        )

    def test_sha256_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            sha256_file(self.tmp_path / "nonexistent.txt")


class GitProvenanceTests(unittest.TestCase):
    def test_git_provenance_in_repo_succeeds(self) -> None:
        # Must run inside C:/Real-Time-Exam-Proctoring-System which is a git repo
        prov = get_git_provenance(REPO_ROOT)
        self.assertIsInstance(prov["commit"], str)
        self.assertNotEqual(prov["commit"].upper(), "UNKNOWN")
        self.assertIsInstance(prov["branch"], str)
        self.assertNotEqual(prov["branch"].upper(), "UNKNOWN")
        self.assertIsInstance(prov["dirty_files"], list)

    def test_unknown_commit_rejected_by_validation(self) -> None:
        """validate() must reject UNKNOWN even if git succeeded somewhere else."""
        from pose_gaze.holistic.debug.session_manifest import SessionManifest
        m = SessionManifest(
            session_id="s", command="c", working_directory="/", 
            git={"commit": "UNKNOWN", "branch": "main"},
            model_artifacts={"m.json": "a" * 64},
            runtime_arguments={"x": 1}, camera_config={"index": 0},
            performance_metrics={}, wall_clock_start="2026-01-01T00:00:00Z",
            wall_clock_end="2026-01-01T00:01:00Z",
        )
        with self.assertRaisesRegex(ValueError, "git.commit"):
            m.validate()



class ModelArtifactHashTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_hashes_json_and_ubj_only(self) -> None:
        (self.tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
        (self.tmp_path / "model.ubj").write_bytes(b"\x00\x01\x02")
        (self.tmp_path / "readme.txt").write_text("ignore", encoding="utf-8")
        hashes = compute_model_artifacts_hashes(self.tmp_path)
        self.assertIn("metrics.json", hashes)
        self.assertIn("model.ubj", hashes)
        self.assertNotIn("readme.txt", hashes)

    def test_verify_mismatch_raises(self) -> None:
        (self.tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
        stored = {"metrics.json": "deadbeef" * 8}
        with self.assertRaisesRegex(ValueError, "HASH MISMATCH"):
            verify_model_artifacts(self.tmp_path, stored)


class SessionManifestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _valid_manifest(self) -> SessionManifest:
        return SessionManifest(
            session_id="test_session",
            command="python -m pose_gaze.holistic.test_webcam --camera 0",
            working_directory=str(REPO_ROOT),
            git={"commit": "abc123def456", "branch": "new-test-pipieline", "is_dirty": False, "dirty_files": []},
            model_artifacts={"causal_actor_metrics.json": "a" * 64},
            runtime_arguments={"camera": 0, "target_fps": 30},
            camera_config={"index": 0, "width": 640, "height": 480, "target_fps": 30},
            performance_metrics={"frame_count": 100},
            wall_clock_start="2026-08-18T00:00:00+00:00",
            wall_clock_end="2026-08-18T00:01:00+00:00",
        )

    def test_valid_manifest_passes(self) -> None:
        m = self._valid_manifest()
        m.validate()  # must not raise

    def test_unknown_git_commit_rejected(self) -> None:
        m = self._valid_manifest()
        m.git["commit"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "git.commit"):
            m.validate()

    def test_unknown_git_branch_rejected(self) -> None:
        m = self._valid_manifest()
        m.git["branch"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "git.branch"):
            m.validate()

    def test_empty_runtime_arguments_rejected(self) -> None:
        m = self._valid_manifest()
        m.runtime_arguments = {}
        with self.assertRaisesRegex(ValueError, "runtime_arguments"):
            m.validate()

    def test_empty_camera_config_rejected(self) -> None:
        m = self._valid_manifest()
        m.camera_config = {}
        with self.assertRaisesRegex(ValueError, "camera_config"):
            m.validate()

    def test_empty_model_artifacts_rejected(self) -> None:
        m = self._valid_manifest()
        m.model_artifacts = {}
        with self.assertRaisesRegex(ValueError, "model_artifacts"):
            m.validate()

    def test_save_load_round_trip(self) -> None:
        m = self._valid_manifest()
        path = self.tmp_path / "manifest.json"
        m.save(path)
        loaded = SessionManifest.load(path)
        self.assertEqual(loaded.session_id, "test_session")
        self.assertEqual(loaded.git["commit"], "abc123def456")


class SessionManifestRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.model_dir = self.tmp_path / "model"
        self.model_dir.mkdir()
        (self.model_dir / "causal_actor_metrics.json").write_text("{}", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_recorder_requires_runtime_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime_arguments"):
            SessionManifestRecorder(
                session_id="s1",
                working_directory=REPO_ROOT,
                model_dir=self.model_dir,
                runtime_arguments={},  # empty → error
                camera_config={"index": 0},
            )

    def test_recorder_requires_camera_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera_config"):
            SessionManifestRecorder(
                session_id="s1",
                working_directory=REPO_ROOT,
                model_dir=self.model_dir,
                runtime_arguments={"camera": 0},
                camera_config={},  # empty → error
            )

    def test_latency_stats_correct(self) -> None:
        recorder = SessionManifestRecorder(
            session_id="s1",
            working_directory=REPO_ROOT,
            model_dir=self.model_dir,
            runtime_arguments={"camera": 0},
            camera_config={"index": 0, "width": 640, "height": 480, "target_fps": 30},
        )
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            recorder.record_frame_latency(v)
        metrics = recorder.compute_performance_metrics()
        self.assertEqual(metrics["frame_count"], 5)
        self.assertEqual(metrics["latency_ms_p50"], 30.0)
        self.assertEqual(metrics["latency_ms_p95"], 50.0)


if __name__ == "__main__":
    unittest.main()
