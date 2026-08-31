"""Session manifest provenance recorder for live and replay sessions.

Validation rules:
- git commit/branch must not be "UNKNOWN" (git must succeed)
- runtime_arguments and camera_config must be explicitly provided (non-empty)
- model_artifacts must contain at least the metrics file hash
- video_path and trace_path, once set, must be non-empty strings

Replay verification:
- verify_against() hashes a fresh model_dir and compares to stored hashes,
  rejecting any mismatch so replay cannot use a different artifact than capture.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path | str) -> str:
    """Compute hex SHA256 hash of a file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hashing: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_provenance(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Capture current git commit, branch, and dirty file list.

    Raises RuntimeError if git is not available or the directory is not a
    git repo.  Callers must handle this rather than silently record UNKNOWN.
    """
    cwd = str(repo_root) if repo_root else os.getcwd()

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"git rev-parse failed in {cwd}: {exc.stderr.strip()}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError("git is not installed or not on PATH") from exc

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"git rev-parse branch failed in {cwd}: {exc.stderr.strip()}"
        ) from exc

    try:
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"git status failed in {cwd}: {exc.stderr.strip()}"
        ) from exc

    dirty_files = [
        line.strip()
        for line in status_output.splitlines()
        if line.strip()
    ]

    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": len(dirty_files) > 0,
        "dirty_files": dirty_files,
    }


def compute_model_artifacts_hashes(model_dir: Path | str) -> dict[str, str]:
    """SHA256 all metrics/schema/model files in model_dir."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    hashes: dict[str, str] = {}
    for ext in ("*.json", "*.ubj", "*.pkl", "*.pt"):
        for fp in sorted(model_dir.rglob(ext)):
            if fp.is_file():
                hashes[fp.relative_to(model_dir).as_posix()] = sha256_file(fp)
    return hashes


def verify_model_artifacts(
    model_dir: Path | str,
    stored_hashes: dict[str, str],
) -> None:
    """Recompute hashes and compare against stored_hashes.

    Raises ValueError listing every mismatch so replay cannot silently run
    against a different artifact than the one used during capture.
    """
    current = compute_model_artifacts_hashes(model_dir)
    errors: list[str] = []
    for name, stored_hash in stored_hashes.items():
        if name not in current:
            errors.append(f"  MISSING artifact file: {name}")
        elif current[name] != stored_hash:
            errors.append(
                f"  HASH MISMATCH {name}: "
                f"stored={stored_hash[:12]}… current={current[name][:12]}…"
            )
    for name in current:
        if name not in stored_hashes:
            errors.append(f"  NEW artifact file not in manifest: {name}")
    if errors:
        raise ValueError(
            "Model artifact provenance mismatch — "
            "replay artifact differs from capture:\n" + "\n".join(errors)
        )


@dataclass
class SessionManifest:
    """Provenance record for a camera or replay session."""

    session_id: str
    command: str
    working_directory: str
    git: dict[str, Any]
    model_artifacts: dict[str, str]
    runtime_arguments: dict[str, Any]
    camera_config: dict[str, Any]
    performance_metrics: dict[str, Any]
    wall_clock_start: str
    wall_clock_end: str
    video_path: str | None = None
    trace_path: str | None = None
    bundle_provenance: dict[str, Any] | None = None
    compute_switches: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> Path:
        self.validate()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, path: Path | str) -> SessionManifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)

    def validate(self) -> None:
        """Reject any manifest with missing or UNKNOWN mandatory fields."""
        mandatory = [
            ("session_id", self.session_id),
            ("command", self.command),
            ("working_directory", self.working_directory),
            ("wall_clock_start", self.wall_clock_start),
            ("wall_clock_end", self.wall_clock_end),
        ]
        for name, value in mandatory:
            if not value or str(value).strip() == "":
                raise ValueError(f"Mandatory manifest field is empty: {name}")

        # Git fields must not be the sentinel "UNKNOWN"
        commit = (self.git or {}).get("commit", "")
        branch = (self.git or {}).get("branch", "")
        if not commit or commit.upper() == "UNKNOWN":
            raise ValueError(
                "Manifest git.commit is missing or UNKNOWN — "
                "capture must run inside a git repository"
            )
        if not branch or branch.upper() == "UNKNOWN":
            raise ValueError(
                "Manifest git.branch is missing or UNKNOWN — "
                "capture must run inside a git repository"
            )

        # Model artifacts required
        model_unavailable = bool(
            (self.runtime_arguments or {}).get("model_unavailable_error")
        )
        if (
            not isinstance(self.model_artifacts, dict)
            or (not self.model_artifacts and not model_unavailable)
        ):
            raise ValueError(
                "Manifest requires non-empty model_artifacts hashes"
            )

        # runtime_arguments must be explicitly populated
        if not isinstance(self.runtime_arguments, dict) or not self.runtime_arguments:
            raise ValueError(
                "Manifest runtime_arguments cannot be empty — "
                "pass all CLI flags used during capture"
            )

        # camera_config must be explicitly populated
        if not isinstance(self.camera_config, dict) or not self.camera_config:
            raise ValueError(
                "Manifest camera_config cannot be empty — "
                "record camera index, width, height, target_fps"
            )

        if not isinstance(self.performance_metrics, dict):
            raise ValueError("Manifest performance_metrics must be a dict")

    def verify_replay_artifact(self, model_dir: Path | str) -> None:
        """Hash model_dir and compare against manifest.model_artifacts.

        Call this at the start of every replay to ensure artifact identity.
        """
        verify_model_artifacts(model_dir, self.model_artifacts)


class SessionManifestRecorder:
    """Collect provenance and timing across a session lifecycle."""

    def __init__(
        self,
        *,
        session_id: str,
        command: str | None = None,
        working_directory: Path | str | None = None,
        model_dir: Path | str | None = None,
        runtime_arguments: Mapping[str, Any],
        camera_config: Mapping[str, Any],
        video_path: Path | str | None = None,
        trace_path: Path | str | None = None,
    ) -> None:
        if not runtime_arguments:
            raise ValueError(
                "runtime_arguments must be non-empty — pass all CLI flags"
            )
        if not camera_config:
            raise ValueError(
                "camera_config must be non-empty — pass camera index, w, h, fps"
            )
        self.session_id = str(session_id)
        self.command = command or " ".join(sys.argv)
        self.working_directory = str(working_directory or os.getcwd())
        self.model_dir = Path(model_dir) if model_dir else None
        self.runtime_arguments = dict(runtime_arguments)
        self.camera_config = dict(camera_config)
        self.video_path = str(video_path) if video_path else None
        self.trace_path = str(trace_path) if trace_path else None
        self.start_time = datetime.now(timezone.utc).isoformat()
        self._latencies_ms: list[float] = []
        self._end_to_end_latencies_ms: list[float] = []
        self._observation_gaps_ms: list[float] = []
        self._last_observation_timestamp_ms: int | None = None
        self._frame_count = 0
        self._skipped_observation_count = 0
        self.bundle_provenance: dict[str, Any] = {}
        self.compute_switches: list[dict[str, Any]] = []
        self._start_perf_time = time.perf_counter()

    def record_frame_latency(
        self,
        latency_ms: float,
        *,
        end_to_end_latency_ms: float | None = None,
        observation_timestamp_ms: int | None = None,
    ) -> None:
        self._latencies_ms.append(float(latency_ms))
        if end_to_end_latency_ms is not None:
            self._end_to_end_latencies_ms.append(float(end_to_end_latency_ms))
        if observation_timestamp_ms is not None:
            timestamp = int(observation_timestamp_ms)
            if self._last_observation_timestamp_ms is not None:
                self._observation_gaps_ms.append(
                    float(timestamp - self._last_observation_timestamp_ms)
                )
            self._last_observation_timestamp_ms = timestamp
        self._frame_count += 1

    def record_skipped_observation(self) -> None:
        self._skipped_observation_count += 1

    def set_bundle_provenance(self, provenance: Mapping[str, Any]) -> None:
        self.bundle_provenance = dict(provenance)

    def record_compute_switch(
        self,
        *,
        timestamp_ms: int,
        requested_mode: str,
        active_mode: str,
        result: str,
        devices_before: Mapping[str, Any],
        devices_after: Mapping[str, Any],
        error: str | None,
    ) -> None:
        self.compute_switches.append({
            "timestamp_ms": int(timestamp_ms),
            "requested_mode": str(requested_mode),
            "active_mode": str(active_mode),
            "result": str(result),
            "devices_before": dict(devices_before),
            "devices_after": dict(devices_after),
            "error": error,
        })

    def compute_performance_metrics(self) -> dict[str, Any]:
        elapsed = time.perf_counter() - self._start_perf_time
        measured_fps = (self._frame_count / elapsed) if elapsed > 0 else 0.0
        if self._latencies_ms:
            sl = sorted(self._latencies_ms)
            p50 = sl[int(0.50 * len(sl))]
            p95 = sl[min(len(sl) - 1, int(0.95 * len(sl)))]
            mean_lat = sum(sl) / len(sl)
        else:
            p50 = p95 = mean_lat = 0.0
        def distribution(values: list[float]) -> dict[str, float]:
            if not values:
                return {"p50": 0.0, "p95": 0.0, "max": 0.0}
            ordered = sorted(values)
            return {
                "p50": round(ordered[int(0.50 * len(ordered))], 2),
                "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
                "max": round(max(ordered), 2),
            }

        return {
            "frame_count": self._frame_count,
            "processed_observation_count": self._frame_count,
            "skipped_observation_count": self._skipped_observation_count,
            "elapsed_seconds": round(elapsed, 4),
            "measured_fps": round(measured_fps, 2),
            "latency_ms_p50": round(p50, 2),
            "latency_ms_p95": round(p95, 2),
            "latency_ms_mean": round(mean_lat, 2),
            "observation_gap_ms": distribution(self._observation_gaps_ms),
            "end_to_end_latency_ms": distribution(self._end_to_end_latencies_ms),
        }

    def build_manifest(self, end_time: str | None = None) -> SessionManifest:
        model_hashes: dict[str, str] = {}
        if self.model_dir and self.model_dir.is_dir():
            model_hashes = compute_model_artifacts_hashes(self.model_dir)
        for specialist, digest in self.bundle_provenance.get("models", {}).items():
            model_hashes[f"models/{specialist}.ubj"] = str(digest).lower()
        git_info = get_git_provenance(self.working_directory)
        metrics = self.compute_performance_metrics()
        return SessionManifest(
            session_id=self.session_id,
            command=self.command,
            working_directory=self.working_directory,
            git=git_info,
            model_artifacts=model_hashes,
            runtime_arguments=self.runtime_arguments,
            camera_config=self.camera_config,
            performance_metrics=metrics,
            wall_clock_start=self.start_time,
            wall_clock_end=end_time or datetime.now(timezone.utc).isoformat(),
            video_path=self.video_path,
            trace_path=self.trace_path,
            bundle_provenance=dict(self.bundle_provenance),
            compute_switches=list(self.compute_switches),
        )
