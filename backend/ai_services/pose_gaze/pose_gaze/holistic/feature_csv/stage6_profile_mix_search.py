"""Select per-specialist Stage 6 model profiles using grouped OOF only."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import stage6_retrain as stage6


def _candidate(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=BUNDLE_DIR")
    return name, Path(path)


def _load_scores(
    name: str,
    bundle_dir: Path,
    train_rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    path = bundle_dir / "grouped_oof_input.npz"
    with np.load(path, allow_pickle=False) as data:
        expected_video = np.asarray([str(row["video"]) for row in train_rows])
        expected_actor = np.asarray([str(row["actor_id"]) for row in train_rows])
        expected_sample = np.asarray([int(row["sample_index"]) for row in train_rows], dtype=np.int32)
        if not np.array_equal(data["video"].astype(str), expected_video):
            raise ValueError(f"{name} OOF video rows do not match current 8 FPS input")
        if not np.array_equal(data["actor_id"].astype(str), expected_actor):
            raise ValueError(f"{name} OOF actor rows do not match current 8 FPS input")
        if not np.array_equal(data["sample_index"], expected_sample):
            raise ValueError(f"{name} OOF sample rows do not match current 8 FPS input")
        return {
            specialist: np.asarray(data[f"score_{specialist}"], dtype=np.float32)
            for specialist in stage6.SPECIALISTS
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=_candidate, required=True)
    parser.add_argument("--global-trials", type=int, default=5000)
    parser.add_argument("--search-seed", action="append", type=int)
    parser.add_argument("--c2-profile")
    parser.add_argument("--c3-profile")
    parser.add_argument("--suspicious-profile")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate) < 2:
        raise ValueError("profile mix search requires at least two candidates")
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")

    train_frames, _ = stage6._prepare_rows(args.input, args.manifest, args.json_root)
    train_rows, _ = stage6._prefix_rows(train_frames)
    candidates = {
        name: _load_scores(name, bundle_dir, train_rows)
        for name, bundle_dir in args.candidate
    }
    initial_gates = stage6.behavior._extended_gate_thresholds(train_rows)
    results = []
    names = sorted(candidates)
    fixed_profiles = (args.c2_profile, args.c3_profile, args.suspicious_profile)
    missing = sorted({name for name in fixed_profiles if name and name not in names})
    if missing:
        raise ValueError(f"unknown fixed profiles: {missing}")
    profile_options = [([fixed] if fixed else names) for fixed in fixed_profiles]
    combinations = product(*profile_options)
    search_seeds = args.search_seed or [20260827]
    for profiles in combinations:
        selected = dict(zip(stage6.SPECIALISTS, profiles, strict=True))
        scores = {
            specialist: candidates[selected[specialist]][specialist]
            for specialist in stage6.SPECIALISTS
        }
        for search_seed in search_seeds:
            thresholds, gates, trace = stage6._calibrate_joint_thresholds(
                train_rows,
                scores,
                initial_gates,
                global_trials=args.global_trials,
                search_seed=search_seed,
            )
            results.append({
                "profiles": selected,
                "search_seed": search_seed,
                "actor_macro_f1": trace[-1]["actor_macro_f1"],
                "thresholds": thresholds,
                "gate_thresholds": gates,
                "exact_causal_parity": trace[-1].get("exact_causal_parity") is True,
            })
            print(json.dumps(results[-1], separators=(",", ":")), flush=True)
    results.sort(key=lambda row: row["actor_macro_f1"], reverse=True)
    payload = {
        "selection_data": "grouped OOF training prefixes only",
        "locked_test_read_or_scored": False,
        "specialists": list(stage6.SPECIALISTS),
        "candidate_names": names,
        "combination_count": len(results),
        "global_trials_per_combination": args.global_trials,
        "search_seeds": search_seeds,
        "best": results[0],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "best": results[0]}, indent=2))


if __name__ == "__main__":
    main()
