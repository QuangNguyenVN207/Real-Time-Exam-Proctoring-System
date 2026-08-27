"""Strict loader and grouped-OOF reproduction verifier for Stage 6 bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xgboost as xgb


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def load_stage6_bundle(
    bundle_dir: Path,
    *,
    expected_feature_schemas: Mapping[str, list[str]] | None = None,
    expected_temporal_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load bundle only after file, schema, and temporal-policy contracts match."""

    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Stage 6 bundle manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != "causal_8fps_stage6_v1":
        raise ValueError("unsupported Stage 6 bundle format")

    for relative, expected_hash in manifest.get("file_sha256", {}).items():
        path = bundle_dir / relative
        if not path.is_file():
            raise ValueError(f"Stage 6 bundle file missing: {relative}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Stage 6 bundle hash mismatch for {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    schemas = json.loads((bundle_dir / "feature_schemas.json").read_text(encoding="utf-8"))
    policy = json.loads((bundle_dir / "temporal_policy.json").read_text(encoding="utf-8"))
    if _canonical_hash(schemas) != manifest.get("feature_schema_hash"):
        raise ValueError("Stage 6 feature-schema hash mismatch")
    if _canonical_hash(policy) != manifest.get("temporal_policy_hash"):
        raise ValueError("Stage 6 temporal-policy hash mismatch")
    if expected_feature_schemas is not None and dict(expected_feature_schemas) != schemas:
        raise ValueError("Stage 6 feature schema does not match runtime schema")
    if expected_temporal_policy is not None and dict(expected_temporal_policy) != policy:
        raise ValueError("Stage 6 temporal policy does not match runtime policy")

    return {"manifest": manifest, "feature_schemas": schemas, "temporal_policy": policy}


def verify_grouped_oof_reproduction(bundle_dir: Path) -> dict[str, Any]:
    """Re-run every saved fold model against its saved 8 FPS OOF input."""

    bundle_dir = Path(bundle_dir)
    loaded = load_stage6_bundle(bundle_dir)
    with np.load(bundle_dir / "grouped_oof_input.npz", allow_pickle=False) as source:
        fold_ids = source["fold_id"].astype(str)
        results: dict[str, Any] = {}
        for specialist, names in loaded["feature_schemas"].items():
            matrix = np.asarray(source[f"X_{specialist}"], dtype=np.float32)
            expected = np.asarray(source[f"score_{specialist}"], dtype=np.float32)
            reproduced = np.full(expected.shape, np.nan, dtype=np.float32)
            for fold_id in sorted(set(fold_ids)):
                indices = np.flatnonzero(fold_ids == fold_id)
                model_path = bundle_dir / "fold_models" / fold_id / f"{specialist}.ubj"
                model = xgb.Booster()
                model.load_model(str(model_path))
                model.feature_names = list(names)
                reproduced[indices] = model.predict(
                    xgb.DMatrix(matrix[indices], feature_names=list(names))
                ).astype(np.float32)
            max_error = float(np.nanmax(np.abs(reproduced - expected)))
            if not np.allclose(reproduced, expected, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"grouped OOF reproduction mismatch for {specialist}: max_abs_error={max_error}"
                )
            results[specialist] = {
                "rows": int(len(expected)),
                "feature_count": int(matrix.shape[1]),
                "float32_atol": 1e-6,
                "max_abs_error": max_error,
                "within_float32_tolerance": True,
            }
    return {"verdict": "PASS", "specialists": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_grouped_oof_reproduction(args.bundle_dir), indent=2))


if __name__ == "__main__":
    main()
