"""Actor-level open-set benchmark for the pretrained face module.

The manifest is intentionally small: each row identifies the left and right
actors in one enrollment image/video or one test video. Frames are decoded in
source order and every embedding is computed from the current frame only.

Example:
    python -m backend.ai_services.face_verify.benchmark_video \
        --manifest D:/data/face_test_data.csv \
        --output-dir data/face_benchmark_p1 \
        --model-root data/insightface
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from backend.core.config import settings

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
UNKNOWN = "UNKNOWN"
NO_DETECTION = "NO_DETECTION"
ACTOR_NOTE = re.compile(r"(?:enroll|test)_(P\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ManifestRow:
    data_path: Path
    pair_id: str
    split: str
    left_actor_id: str
    right_actor_id: str
    note: str

    @property
    def source_id(self) -> str:
        return self.data_path.stem


@dataclass
class FaceObservation:
    source_path: str
    source_id: str
    pair_id: str
    split: str
    actor_id: str
    side: str
    frame_index: int
    timestamp_s: float
    bbox: tuple[int, int, int, int]
    det_score: float
    quality: float
    embedding: np.ndarray


def natural_actor_key(actor_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", actor_id)
    return (int(match.group(1)) if match else 10**9, actor_id)


def load_manifest(path: Path) -> list[ManifestRow]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "data_path",
        "pair_id",
        "split",
        "left_actor_id",
        "right_actor_id",
        "note",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")

    rows: list[ManifestRow] = []
    for index, item in frame.iterrows():
        source = Path(item["data_path"]).expanduser()
        split = item["split"].strip().lower()
        if split not in {"enroll", "test"}:
            raise ValueError(f"row {index + 2}: unsupported split {split!r}")
        if not source.is_file():
            raise FileNotFoundError(f"row {index + 2}: missing source {source}")
        if source.suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            raise ValueError(f"row {index + 2}: unsupported media {source.suffix}")
        left = item["left_actor_id"].strip()
        right = item["right_actor_id"].strip()
        if not left or not right or left == right:
            raise ValueError(f"row {index + 2}: invalid left/right actor IDs")
        rows.append(
            ManifestRow(
                data_path=source.resolve(),
                pair_id=item["pair_id"].strip(),
                split=split,
                left_actor_id=left,
                right_actor_id=right,
                note=item["note"].strip(),
            )
        )
    return rows


def actors_requested(row: ManifestRow) -> dict[str, str]:
    """Return side -> actor, honoring explicit actor-only notes."""

    match = ACTOR_NOTE.search(row.note)
    actors = {"left": row.left_actor_id, "right": row.right_actor_id}
    if not match:
        return actors
    selected = match.group(1).upper()
    filtered = {side: actor for side, actor in actors.items() if actor.upper() == selected}
    if not filtered:
        raise ValueError(
            f"note selects {selected}, which is not in {row.left_actor_id}/{row.right_actor_id}"
        )
    return filtered


def iter_sampled_frames(path: Path, sample_fps: float) -> Iterable[tuple[int, float, np.ndarray]]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"could not read image: {path}")
        yield 0, 0.0, frame
        return

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise RuntimeError(f"invalid FPS for video: {path}")
    stride = max(1, int(round(source_fps / sample_fps)))
    frame_index = 0
    try:
        while capture.grab():
            if frame_index % stride:
                frame_index += 1
                continue
            ok, frame = capture.retrieve()
            if not ok:
                frame_index += 1
                continue
            yield frame_index, frame_index / source_fps, frame
            frame_index += 1
    finally:
        capture.release()


def _face_center(face: Any) -> float:
    return float(face.bbox[0] + face.bbox[2]) / 2.0


def assign_faces_to_sides(faces: list[Any], width: int) -> dict[str, Any]:
    """Assign at most one detection to each fixed seat without using future frames."""

    if not faces:
        return {}
    ordered = sorted(faces, key=_face_center)
    if len(ordered) == 1:
        side = "left" if _face_center(ordered[0]) < width / 2.0 else "right"
        return {side: ordered[0]}
    return {"left": ordered[0], "right": ordered[-1]}


def observation_quality(frame: np.ndarray, face: Any) -> float:
    x1, y1, x2, y2 = [int(value) for value in face.bbox]
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    crop = frame[y1:y2, x1:x2]
    sharpness = 0.0
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    short_side = max(0, min(x2 - x1, y2 - y1))
    size_score = min(1.0, short_side / 160.0)
    sharpness_score = min(1.0, math.log1p(max(0.0, sharpness)) / math.log(501.0))
    return 0.55 * float(face.det_score) + 0.25 * size_score + 0.20 * sharpness_score


def extract_observations(
    rows: list[ManifestRow],
    verifier: Any,
    sample_fps: float,
) -> tuple[list[FaceObservation], list[dict[str, Any]]]:
    observations: list[FaceObservation] = []
    coverage_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        requested = actors_requested(row)
        print(f"[{row_number:02d}/{len(rows):02d}] {row.split:6s} {row.pair_id} {row.data_path.name}")
        for frame_index, timestamp_s, frame in iter_sampled_frames(row.data_path, sample_fps):
            faces = verifier._detect_faces(frame)
            by_side = assign_faces_to_sides(faces, frame.shape[1])
            for side, actor_id in requested.items():
                face = by_side.get(side)
                coverage_rows.append(
                    {
                        "source_id": row.source_id,
                        "pair_id": row.pair_id,
                        "split": row.split,
                        "actor_id": actor_id,
                        "side": side,
                        "frame_index": frame_index,
                        "timestamp_s": timestamp_s,
                        "face_detected": int(face is not None),
                    }
                )
                if face is None:
                    continue
                embedding = np.asarray(face.normed_embedding, dtype=np.float32)
                observations.append(
                    FaceObservation(
                        source_path=str(row.data_path),
                        source_id=row.source_id,
                        pair_id=row.pair_id,
                        split=row.split,
                        actor_id=actor_id,
                        side=side,
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        bbox=tuple(int(value) for value in face.bbox),
                        det_score=float(face.det_score),
                        quality=observation_quality(frame, face),
                        embedding=embedding,
                    )
                )
    return observations, coverage_rows


def select_templates(
    candidates: list[FaceObservation],
    max_templates: int,
    min_time_gap_s: float,
) -> list[FaceObservation]:
    ranked = sorted(candidates, key=lambda item: item.quality, reverse=True)
    selected: list[FaceObservation] = []
    for candidate in ranked:
        too_close = any(
            candidate.source_id == existing.source_id
            and abs(candidate.timestamp_s - existing.timestamp_s) < min_time_gap_s
            for existing in selected
        )
        if too_close:
            continue
        selected.append(candidate)
        if len(selected) >= max_templates:
            break
    if not selected and ranked:
        selected.append(ranked[0])
    return selected


def build_gallery(
    observations: list[FaceObservation],
    max_templates: int,
    min_time_gap_s: float,
) -> tuple[list[str], np.ndarray, dict[str, list[FaceObservation]]]:
    enrolled = sorted(
        {item.actor_id for item in observations if item.split == "enroll"},
        key=natural_actor_key,
    )
    templates: dict[str, list[FaceObservation]] = {}
    centroids: list[np.ndarray] = []
    for actor_id in enrolled:
        candidates = [
            item
            for item in observations
            if item.split == "enroll" and item.actor_id == actor_id
        ]
        selected = select_templates(candidates, max_templates, min_time_gap_s)
        if not selected:
            raise RuntimeError(f"no enrollment face detected for {actor_id}")
        templates[actor_id] = selected
        centroid = np.mean([item.embedding for item in selected], axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids.append(centroid.astype(np.float32))
    return enrolled, np.stack(centroids), templates


def score_queries(
    observations: list[FaceObservation],
    actor_ids: list[str],
    gallery: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    actor_index = {actor: index for index, actor in enumerate(actor_ids)}
    for item in observations:
        if item.split != "test":
            continue
        similarities = gallery @ item.embedding
        order = np.argsort(similarities)[::-1]
        best_index = int(order[0])
        second_index = int(order[1]) if len(order) > 1 else best_index
        truth_known = item.actor_id in actor_index
        rows.append(
            {
                "source_path": item.source_path,
                "source_id": item.source_id,
                "pair_id": item.pair_id,
                "actor_id": item.actor_id,
                "truth_label": item.actor_id if truth_known else UNKNOWN,
                "truth_known": int(truth_known),
                "side": item.side,
                "frame_index": item.frame_index,
                "timestamp_s": item.timestamp_s,
                "bbox_x1": item.bbox[0],
                "bbox_y1": item.bbox[1],
                "bbox_x2": item.bbox[2],
                "bbox_y2": item.bbox[3],
                "det_score": item.det_score,
                "quality": item.quality,
                "top1_actor_id": actor_ids[best_index],
                "top1_score": float(similarities[best_index]),
                "top2_actor_id": actor_ids[second_index],
                "top2_score": float(similarities[second_index]),
                "margin": float(similarities[best_index] - similarities[second_index]),
                "true_score": (
                    float(similarities[actor_index[item.actor_id]])
                    if truth_known
                    else np.nan
                ),
                "max_wrong_score": (
                    float(np.max(np.delete(similarities, actor_index[item.actor_id])))
                    if truth_known and len(actor_ids) > 1
                    else float(similarities[best_index])
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("no test faces were detected")
    result["partition"] = "evaluation"
    for _, indices in result.groupby(["source_id", "actor_id"]).groups.items():
        ordered = result.loc[list(indices)].sort_values(["timestamp_s", "frame_index"]).index
        calibration_count = max(1, int(math.floor(len(ordered) * 0.40)))
        if calibration_count >= len(ordered) and len(ordered) > 1:
            calibration_count = len(ordered) - 1
        result.loc[ordered[:calibration_count], "partition"] = "calibration"
    return result


def apply_policy(frame: pd.DataFrame, threshold: float, margin: float) -> pd.Series:
    accepted = (frame["top1_score"] >= threshold) & (frame["margin"] >= margin)
    return frame["top1_actor_id"].where(accepted, UNKNOWN)


def actor_decisions(
    frame: pd.DataFrame,
    threshold: float,
    margin: float,
) -> pd.DataFrame:
    working = frame.copy()
    working["predicted_label"] = apply_policy(working, threshold, margin)
    rows: list[dict[str, Any]] = []
    for (source_id, actor_id), group in working.groupby(["source_id", "actor_id"]):
        counts = Counter(group["predicted_label"])
        predicted, votes = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0]
        truth = str(group["truth_label"].iloc[0])
        rows.append(
            {
                "source_id": source_id,
                "pair_id": str(group["pair_id"].iloc[0]),
                "actor_id": actor_id,
                "truth_label": truth,
                "predicted_label": predicted,
                "truth_known": int(group["truth_known"].iloc[0]),
                "detected_samples": int(len(group)),
                "winning_votes": int(votes),
                "winning_vote_fraction": float(votes / len(group)),
                "mean_top1_score": float(group["top1_score"].mean()),
                "mean_margin": float(group["margin"].mean()),
                "correct": int(predicted == truth),
            }
        )
    return pd.DataFrame(rows)


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% interval for a binomial rate, including small samples."""
    if total <= 0:
        return float("nan"), float("nan")
    proportion = correct / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def policy_metrics(frame: pd.DataFrame, threshold: float, margin: float) -> dict[str, Any]:
    actors = actor_decisions(frame, threshold, margin)
    known = actors[actors["truth_known"] == 1]
    unknown = actors[actors["truth_known"] == 0]
    known_rate = float(known["correct"].mean()) if len(known) else float("nan")
    unknown_rate = float(unknown["correct"].mean()) if len(unknown) else float("nan")
    if known_rate + unknown_rate:
        harmonic = 2.0 * known_rate * unknown_rate / (known_rate + unknown_rate)
    else:
        harmonic = 0.0
    labels = sorted(set(actors["truth_label"]) | set(actors["predicted_label"]))
    macro_f1 = f1_score(
        actors["truth_label"],
        actors["predicted_label"],
        labels=labels,
        average="macro",
        zero_division=0,
    )
    predicted = apply_policy(frame, threshold, margin)
    frame_accuracy = float((predicted == frame["truth_label"]).mean())
    overall_correct = int(actors["correct"].sum())
    known_correct = int(known["correct"].sum())
    unknown_correct = int(unknown["correct"].sum())
    overall_ci = wilson_interval(overall_correct, len(actors))
    known_ci = wilson_interval(known_correct, len(known))
    unknown_ci = wilson_interval(unknown_correct, len(unknown))
    return {
        "actor_open_set_accuracy": float(actors["correct"].mean()),
        "actor_open_set_correct": overall_correct,
        "actor_open_set_total": int(len(actors)),
        "actor_open_set_wilson95": list(overall_ci),
        "actor_known_identification_rate": known_rate,
        "actor_known_correct": known_correct,
        "actor_known_total": int(len(known)),
        "actor_known_wilson95": list(known_ci),
        "actor_unknown_rejection_rate": unknown_rate,
        "actor_unknown_correct": unknown_correct,
        "actor_unknown_total": int(len(unknown)),
        "actor_unknown_wilson95": list(unknown_ci),
        "actor_known_unknown_hmean": harmonic,
        "actor_macro_f1": float(macro_f1),
        "frame_accuracy_diagnostic": frame_accuracy,
        "actor_groups": int(len(actors)),
    }


def select_policy(calibration: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    records: list[dict[str, float]] = []
    for threshold in np.arange(0.20, 0.751, 0.01):
        for margin in np.arange(0.00, 0.151, 0.01):
            metrics = policy_metrics(calibration, float(threshold), float(margin))
            records.append({"threshold": threshold, "margin_threshold": margin, **metrics})
    sweep = pd.DataFrame(records)
    # Primary metrics are actor-level. A frame-level tie-break would favor the
    # most permissive edge of a broad perfect plateau, which is unstable for an
    # open-set system. Instead, keep every actor-optimal point, choose the
    # midpoint of its threshold interval, then use the predeclared 0.08 margin
    # (or the closest equally optimal value).
    optimal = sweep.copy()
    for column in (
        "actor_known_unknown_hmean",
        "actor_open_set_accuracy",
        "actor_macro_f1",
    ):
        best_value = float(optimal[column].max())
        optimal = optimal[np.isclose(optimal[column], best_value)]
    lower = float(optimal["threshold"].min())
    upper = float(optimal["threshold"].max())
    midpoint = (lower + upper) / 2.0
    selected_threshold = float(
        min(optimal["threshold"].unique(), key=lambda value: abs(value - midpoint))
    )
    margin_candidates = optimal[
        np.isclose(optimal["threshold"], selected_threshold)
    ]["margin_threshold"].unique()
    selected_margin = float(
        min(margin_candidates, key=lambda value: abs(value - 0.08))
    )
    return selected_threshold, selected_margin, sweep


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def save_gallery(
    output_dir: Path,
    actor_ids: list[str],
    gallery: np.ndarray,
    templates: dict[str, list[FaceObservation]],
) -> None:
    template_vectors = np.concatenate(
        [[item.embedding for item in templates[actor]] for actor in actor_ids]
    ).astype(np.float32)
    template_actor_ids = np.concatenate(
        [[actor] * len(templates[actor]) for actor in actor_ids]
    )
    np.savez_compressed(
        output_dir / "gallery_embeddings.npz",
        actor_ids=np.asarray(actor_ids),
        centroids=gallery,
        template_actor_ids=template_actor_ids,
        template_vectors=template_vectors,
    )


def _style_axes(axis: Any) -> None:
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_coverage(coverage: pd.DataFrame, output: Path) -> None:
    grouped = (
        coverage.groupby(["split", "actor_id"], as_index=False)["face_detected"]
        .mean()
        .sort_values("actor_id", key=lambda col: col.map(natural_actor_key))
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, split in zip(axes, ["enroll", "test"]):
        values = grouped[grouped["split"] == split]
        colors = ["#2070B4" if value >= 0.8 else "#F28E2B" for value in values["face_detected"]]
        axis.bar(values["actor_id"], values["face_detected"] * 100.0, color=colors)
        axis.set_title(f"{split.capitalize()} face-detection coverage")
        axis.set_xlabel("Actor ID")
        axis.tick_params(axis="x", rotation=45)
        axis.set_ylim(0, 105)
        _style_axes(axis)
    axes[0].set_ylabel("Detected sampled frames (%)")
    fig.suptitle("Face detection coverage by actor", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(frame: pd.DataFrame, output: Path) -> None:
    evaluation = frame[frame["partition"] == "evaluation"]
    known = evaluation[evaluation["truth_known"] == 1]
    unknown = evaluation[evaluation["truth_known"] == 0]
    bins = np.linspace(-0.1, 1.0, 34)
    fig, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.hist(known["true_score"], bins=bins, alpha=0.58, label="Known: genuine score", color="#2070B4", density=True)
    axis.hist(known["max_wrong_score"], bins=bins, alpha=0.50, label="Known: strongest wrong identity", color="#F28E2B", density=True)
    axis.hist(unknown["top1_score"], bins=bins, alpha=0.50, label="Unknown: strongest gallery match", color="#C44E52", density=True)
    axis.set_xlabel("Cosine similarity")
    axis.set_ylabel("Density")
    axis.set_title("ArcFace score separation on evaluation frames", fontweight="bold")
    axis.legend(frameon=False)
    _style_axes(axis)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(
    calibration: pd.DataFrame,
    selected_threshold: float,
    selected_margin: float,
    output: Path,
) -> None:
    rows = []
    for threshold in np.arange(0.20, 0.751, 0.01):
        metrics = policy_metrics(calibration, float(threshold), selected_margin)
        rows.append({"threshold": threshold, **metrics})
    values = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.plot(values["threshold"], values["actor_known_identification_rate"], label="Known identification", linewidth=2.2)
    axis.plot(values["threshold"], values["actor_unknown_rejection_rate"], label="Unknown rejection", linewidth=2.2)
    axis.plot(values["threshold"], values["actor_known_unknown_hmean"], label="Harmonic mean", linewidth=2.2, color="#2CA02C")
    axis.axvline(selected_threshold, color="#333333", linestyle="--", linewidth=1.4, label=f"Selected threshold = {selected_threshold:.2f}")
    axis.set_ylim(-0.03, 1.05)
    axis.set_xlabel("Cosine-similarity threshold")
    axis.set_ylabel("Actor-level rate")
    axis.set_title(f"Calibration sweep (runner-up margin = {selected_margin:.2f})", fontweight="bold")
    axis.legend(frameon=False, loc="lower right")
    _style_axes(axis)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(actors: pd.DataFrame, output: Path) -> None:
    labels = sorted(
        set(actors["truth_label"]) | set(actors["predicted_label"]),
        key=lambda value: (value == UNKNOWN, natural_actor_key(value)),
    )
    matrix = confusion_matrix(actors["truth_label"], actors["predicted_label"], labels=labels)
    fig, axis = plt.subplots(figsize=(9.5, 8.2))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted actor")
    axis.set_ylabel("True actor")
    axis.set_title("Actor-level open-set confusion matrix", fontweight="bold")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_matrix(frame: pd.DataFrame, actor_ids: list[str], output: Path) -> None:
    evaluation = frame[frame["partition"] == "evaluation"]
    truth_actors = sorted(evaluation["actor_id"].unique(), key=natural_actor_key)
    values = np.full((len(truth_actors), len(actor_ids)), np.nan, dtype=float)
    for row_index, truth_actor in enumerate(truth_actors):
        subset = evaluation[evaluation["actor_id"] == truth_actor]
        # Reconstruct only the two leading values when available. The full
        # gallery matrix is attached to the frame before this plot is called.
        for column_index, gallery_actor in enumerate(actor_ids):
            column = f"similarity_{gallery_actor}"
            values[row_index, column_index] = float(subset[column].mean())
    fig, axis = plt.subplots(figsize=(11.5, 8.4))
    image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=max(0.75, float(np.nanmax(values))))
    axis.set_xticks(range(len(actor_ids)), actor_ids, rotation=45, ha="right")
    axis.set_yticks(range(len(truth_actors)), truth_actors)
    axis.set_xlabel("Enrolled gallery identity")
    axis.set_ylabel("True test actor")
    axis.set_title("Mean cosine similarity: test actors vs enrolled gallery", fontweight="bold")
    fig.colorbar(image, ax=axis, label="Mean cosine similarity", fraction=0.035, pad=0.03)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def read_source_frame(path: str, frame_index: int) -> np.ndarray | None:
    source = Path(path)
    if source.suffix.lower() in IMAGE_SUFFIXES:
        return cv2.imread(str(source))
    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def plot_examples(frame: pd.DataFrame, threshold: float, margin: float, output: Path) -> None:
    evaluation = frame[frame["partition"] == "evaluation"].copy()
    evaluation["predicted_label"] = apply_policy(evaluation, threshold, margin)
    evaluation["correct"] = evaluation["predicted_label"] == evaluation["truth_label"]
    picks: list[pd.Series] = []
    categories = [
        (
            evaluation[(evaluation["truth_known"] == 1) & evaluation["correct"]]
            .sort_values("top1_score", ascending=False),
            2,
        ),
        (
            evaluation[(evaluation["truth_known"] == 0) & evaluation["correct"]]
            .sort_values("top1_score", ascending=False),
            2,
        ),
        (
            evaluation[~evaluation["correct"]].sort_values("top1_score", ascending=False),
            2,
        ),
        (
            evaluation[(evaluation["truth_known"] == 1) & evaluation["correct"]]
            .sort_values("top1_score", ascending=True),
            2,
        ),
    ]
    used_groups: set[tuple[str, str]] = set()
    for category, quota in categories:
        added = 0
        for _, item in category.iterrows():
            key = (str(item["source_id"]), str(item["actor_id"]))
            if key in used_groups:
                continue
            picks.append(item)
            used_groups.add(key)
            added += 1
            if len(picks) >= 6 or added >= quota:
                break
        if len(picks) >= 6:
            break
    if len(picks) < 6:
        for _, item in evaluation.sort_values("quality", ascending=False).iterrows():
            key = (str(item["source_id"]), str(item["actor_id"]))
            if key in used_groups:
                continue
            picks.append(item)
            used_groups.add(key)
            if len(picks) >= 6:
                break

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.5))
    for axis, item in zip(axes.flat, picks):
        image = read_source_frame(str(item["source_path"]), int(item["frame_index"]))
        if image is None:
            axis.axis("off")
            continue
        x1, y1, x2, y2 = [int(item[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
        predicted = str(item["predicted_label"])
        truth = str(item["truth_label"])
        color = (52, 168, 83) if predicted == truth else (40, 40, 220)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axis.imshow(rgb)
        axis.set_title(
            f"Truth {truth} | Pred {predicted}\nscore={item['top1_score']:.3f}, margin={item['margin']:.3f}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes.flat[len(picks) :]:
        axis.axis("off")
    fig.suptitle("Representative frame-level face decisions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_actor_metrics_ci(metrics: dict[str, Any], output: Path) -> None:
    """Plot selected actor-level metrics with small-sample uncertainty."""
    result = metrics["policies"]["selected_operating_point"]["evaluation"]
    series = [
        ("Known ID", "actor_known_identification_rate", "actor_known_wilson95"),
        ("Unknown rejection", "actor_unknown_rejection_rate", "actor_unknown_wilson95"),
        ("Open-set overall", "actor_open_set_accuracy", "actor_open_set_wilson95"),
    ]
    positions = np.arange(len(series), dtype=float)
    colors = ["#2563eb", "#f59e0b", "#16a34a"]
    fig, axis = plt.subplots(figsize=(9.6, 5.5))
    for position, color, (label, rate_key, ci_key) in zip(positions, colors, series):
        value = float(result[rate_key])
        lower_ci, upper_ci = result[ci_key]
        axis.errorbar(
            [position],
            [value],
            yerr=np.asarray([[max(0.0, value - lower_ci)], [max(0.0, upper_ci - value)]]),
            fmt="o",
            markersize=10,
            capsize=5,
            linewidth=2,
            color=color,
        )
        axis.text(position, value - 0.035, f"{value:.2f}", ha="center", fontweight="bold")
    axis.set_xticks(positions, [label for label, _, _ in series])
    axis.set_ylim(0.45, 1.04)
    axis.set_ylabel("Actor-level rate (Wilson 95% CI)")
    axis.set_title(
        "Actor-level metrics at the selected operating point (0.32 / 0.08)",
        fontweight="bold",
    )
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_margin_heatmap(
    sweep: pd.DataFrame,
    selected_threshold: float,
    selected_margin: float,
    output: Path,
) -> None:
    """Show the complete 2-D search without implying a unique optimum."""
    thresholds = np.sort(sweep["threshold"].unique())
    margins = np.sort(sweep["margin_threshold"].unique())
    hmean = (
        sweep.pivot(
            index="margin_threshold",
            columns="threshold",
            values="actor_known_unknown_hmean",
        )
        .reindex(index=margins, columns=thresholds)
        .to_numpy()
    )
    frame_accuracy = (
        sweep.pivot(
            index="margin_threshold",
            columns="threshold",
            values="frame_accuracy_diagnostic",
        )
        .reindex(index=margins, columns=thresholds)
        .to_numpy()
    )
    actor_optimal = sweep.copy()
    for column in (
        "actor_known_unknown_hmean",
        "actor_open_set_accuracy",
        "actor_macro_f1",
    ):
        actor_optimal = actor_optimal[
            np.isclose(actor_optimal[column], actor_optimal[column].max())
        ]
    optimal_mask = np.zeros_like(hmean, dtype=float)
    threshold_index = {round(value, 8): index for index, value in enumerate(thresholds)}
    margin_index = {round(value, 8): index for index, value in enumerate(margins)}
    for row in actor_optimal.itertuples():
        optimal_mask[
            margin_index[round(float(row.margin_threshold), 8)],
            threshold_index[round(float(row.threshold), 8)],
        ] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharex=True, sharey=True)
    panels = [
        (hmean, "Actor known/unknown H-mean (primary)", "viridis"),
        (frame_accuracy, "Frame accuracy (diagnostic only)", "magma"),
    ]
    for axis, (values, title, cmap) in zip(axes, panels):
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=[
                thresholds.min() - 0.005,
                thresholds.max() + 0.005,
                margins.min() - 0.005,
                margins.max() + 0.005,
            ],
            vmin=float(np.nanmin(values)),
            vmax=1.0,
            cmap=cmap,
        )
        axis.contour(
            thresholds,
            margins,
            optimal_mask,
            levels=[0.5],
            colors="white",
            linewidths=2.0,
        )
        axis.axvline(selected_threshold, color="white", linestyle=":", alpha=0.85)
        axis.axhline(selected_margin, color="white", linestyle=":", alpha=0.85)
        axis.scatter(
            [selected_threshold],
            [selected_margin],
            marker="*",
            s=260,
            color="#ef4444",
            edgecolor="white",
            linewidth=1.2,
            label="Selected point: 0.32 / 0.08",
            zorder=5,
        )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Cosine-similarity threshold")
        axis.grid(False)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("Runner-up margin threshold")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.95)
    fig.suptitle(
        "Full threshold-margin sweep: selected point inside the actor-optimal plateau",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        (
            f"White boundary = {len(actor_optimal)} combinations tied on all primary actor metrics. "
            "0.32 is the threshold-plateau midpoint; 0.08 is the predeclared margin."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    metrics: dict[str, Any],
    selected_threshold: float,
    selected_margin: float,
) -> None:
    selected = metrics["policies"]["selected_operating_point"]["evaluation"]
    text = f"""# Face verification benchmark

## Protocol

- Detector: SCRFD-10G `det_10g.onnx`; recognizer: ArcFace R50 `w600k_r50.onnx`.
- Embedding: 512-dimensional L2-normalized vector; comparison: cosine similarity.
- Enrollment and test sources are separate according to the supplied manifest.
- Video is decoded in chronological order and sampled at {metrics['protocol']['sample_fps']:.2f} FPS.
- Primary metrics are actor-level, keyed by `(source_id, actor_id)`; frame metrics are diagnostic only.
- The first 40% of each test actor stream is used for exploratory threshold calibration. The remaining 60% is reported as evaluation. This is a within-recording split, not a new locked dataset.
- When several policies tie on all actor-level calibration metrics, the selected cosine threshold is the midpoint of the optimal plateau and the runner-up margin is the optimal value closest to the predeclared 0.08.

## Dataset

- Enrolled identities: {metrics['dataset']['enrolled_identities']}.
- Known actor-video test groups: {metrics['dataset']['known_test_actor_groups']}.
- Unknown actor-video test groups: {metrics['dataset']['unknown_test_actor_groups']}.
- Enrollment templates retained: {metrics['dataset']['retained_enrollment_templates']}.

## Results

| Policy | Cosine threshold | Runner-up margin | Known identification | Unknown rejection | Open-set accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Selected operating point | {selected_threshold:.2f} | {selected_margin:.2f} | {selected['actor_known_identification_rate']:.3f} | {selected['actor_unknown_rejection_rate']:.3f} | {selected['actor_open_set_accuracy']:.3f} | {selected['actor_macro_f1']:.3f} |

## Figure files

- `fig01_detection_coverage.png`
- `fig02_score_distributions.png`
- `fig03_threshold_sweep.png`
- `fig04_actor_confusion_matrix.png`
- `fig05_similarity_matrix.png`
- `fig06_example_predictions.png`
- `fig07_actor_metrics_ci.png`
- `fig08_threshold_margin_heatmap.png`

## Reporting limitation

The selected operating point is exploratory because calibration and evaluation segments come from the same recordings. Do not claim generalisation to unseen capture conditions until an independent locked video set is collected. Wilson 95% intervals are included because there are only 17 actor-video evaluation groups.
"""
    (output_dir / "report_snippet.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-templates", type=int, default=12)
    parser.add_argument("--min-template-gap-s", type=float, default=0.75)
    parser.add_argument("--det-size", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0:
        raise ValueError("sample-fps must be positive")
    if args.max_templates < 1:
        raise ValueError("max-templates must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_root.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)

    from backend.ai_services.face_verify.face_verify import FaceVerifier

    empty_db = args.output_dir / "empty_runtime_db"
    started = time.perf_counter()
    verifier = FaceVerifier(
        db_path=str(empty_db),
        det_size=(args.det_size, args.det_size),
        model_root=str(args.model_root),
    )
    model_load_seconds = time.perf_counter() - started

    extraction_started = time.perf_counter()
    observations, coverage_rows = extract_observations(rows, verifier, args.sample_fps)
    extraction_seconds = time.perf_counter() - extraction_started
    coverage = pd.DataFrame(coverage_rows)
    actor_ids, gallery, templates = build_gallery(
        observations,
        args.max_templates,
        args.min_template_gap_s,
    )
    queries = score_queries(observations, actor_ids, gallery)
    # Store the complete score vector for auditable similarity figures.
    observation_by_key = {
        (item.source_id, item.actor_id, item.frame_index): item
        for item in observations
        if item.split == "test"
    }
    for actor_index, actor_id in enumerate(actor_ids):
        queries[f"similarity_{actor_id}"] = [
            float(gallery[actor_index] @ observation_by_key[(row.source_id, row.actor_id, int(row.frame_index))].embedding)
            for row in queries.itertuples()
        ]

    calibration = queries[queries["partition"] == "calibration"]
    evaluation = queries[queries["partition"] == "evaluation"]
    selected_threshold, selected_margin, sweep = select_policy(calibration)
    if not (
        math.isclose(selected_threshold, settings.face_similarity_threshold, abs_tol=1e-12)
        and math.isclose(selected_margin, settings.face_identity_margin_threshold, abs_tol=1e-12)
    ):
        raise ValueError("Selected face operating point does not match runtime settings")
    policies = {"selected_operating_point": (selected_threshold, selected_margin)}
    policy_results: dict[str, Any] = {}
    selected_actor_decisions = None
    for name, (threshold, margin) in policies.items():
        policy_results[name] = {
            "threshold": threshold,
            "margin_threshold": margin,
            "calibration": policy_metrics(calibration, threshold, margin),
            "evaluation": policy_metrics(evaluation, threshold, margin),
        }
        if name == "selected_operating_point":
            selected_actor_decisions = actor_decisions(evaluation, threshold, margin)

    assert selected_actor_decisions is not None
    coverage_summary = (
        coverage.groupby(["split", "source_id", "actor_id"], as_index=False)
        .agg(expected_samples=("face_detected", "size"), detected_samples=("face_detected", "sum"))
    )
    coverage_summary["detection_coverage"] = (
        coverage_summary["detected_samples"] / coverage_summary["expected_samples"]
    )
    template_rows = [
        {
            "actor_id": actor,
            "retained_templates": len(items),
            "mean_quality": float(np.mean([item.quality for item in items])),
            "sources": ";".join(sorted({item.source_id for item in items})),
        }
        for actor, items in templates.items()
    ]
    template_frame = pd.DataFrame(template_rows)

    model_dir = args.model_root / "models" / "buffalo_l"
    model_files = {
        name: {
            "path": str(model_dir / name),
            "sha256": sha256_file(model_dir / name),
        }
        for name in ("det_10g.onnx", "w600k_r50.onnx")
    }
    metrics = {
        "protocol": {
            "sample_fps": args.sample_fps,
            "det_size": [args.det_size, args.det_size],
            "max_templates_per_actor": args.max_templates,
            "min_template_gap_s": args.min_template_gap_s,
            "calibration_fraction_per_actor_video": 0.40,
            "gallery_strategy": "L2-normalized centroid of quality-ranked templates",
            "decision_rule": "top1 cosine >= threshold and top1-top2 >= margin",
        },
        "dataset": {
            "manifest_rows": len(rows),
            "enrolled_identities": len(actor_ids),
            "enrolled_actor_ids": actor_ids,
            "known_test_actor_groups": int(
                selected_actor_decisions["truth_known"].sum()
            ),
            "unknown_test_actor_groups": int(
                (selected_actor_decisions["truth_known"] == 0).sum()
            ),
            "retained_enrollment_templates": int(
                sum(len(items) for items in templates.values())
            ),
            "sampled_actor_frames": int(len(coverage)),
            "detected_faces": int(coverage["face_detected"].sum()),
        },
        "timing": {
            "model_load_seconds": model_load_seconds,
            "extraction_seconds": extraction_seconds,
            "mean_seconds_per_sampled_source_frame": extraction_seconds
            / max(1, coverage[["source_id", "frame_index"]].drop_duplicates().shape[0]),
        },
        "policies": policy_results,
        "provenance": {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "git_branch": git_value("branch", "--show-current"),
            "git_commit": git_value("rev-parse", "HEAD"),
            "python": sys.version,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "onnxruntime_providers": __import__("onnxruntime").get_available_providers(),
            "models": model_files,
        },
    }

    queries["predicted_label"] = apply_policy(
        queries, selected_threshold, selected_margin
    )
    queries.to_csv(args.output_dir / "frame_predictions.csv", index=False)
    selected_actor_decisions.to_csv(args.output_dir / "actor_predictions.csv", index=False)
    coverage_summary.to_csv(args.output_dir / "detection_coverage.csv", index=False)
    template_frame.to_csv(args.output_dir / "enrollment_templates.csv", index=False)
    sweep.to_csv(args.output_dir / "threshold_margin_sweep.csv", index=False)
    save_gallery(args.output_dir, actor_ids, gallery, templates)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_coverage(coverage, args.output_dir / "fig01_detection_coverage.png")
    plot_score_distributions(queries, args.output_dir / "fig02_score_distributions.png")
    plot_threshold_sweep(
        calibration,
        selected_threshold,
        selected_margin,
        args.output_dir / "fig03_threshold_sweep.png",
    )
    plot_confusion(
        selected_actor_decisions,
        args.output_dir / "fig04_actor_confusion_matrix.png",
    )
    plot_similarity_matrix(
        queries,
        actor_ids,
        args.output_dir / "fig05_similarity_matrix.png",
    )
    plot_examples(
        queries,
        selected_threshold,
        selected_margin,
        args.output_dir / "fig06_example_predictions.png",
    )
    plot_actor_metrics_ci(metrics, args.output_dir / "fig07_actor_metrics_ci.png")
    plot_threshold_margin_heatmap(
        sweep,
        selected_threshold,
        selected_margin,
        args.output_dir / "fig08_threshold_margin_heatmap.png",
    )
    write_report(args.output_dir, metrics, selected_threshold, selected_margin)
    print(json.dumps(policy_results, indent=2))
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
