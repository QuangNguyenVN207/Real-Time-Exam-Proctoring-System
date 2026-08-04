"""Export XGBoost CSV files and per-class skeleton previews."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .artifact import load_raw_npz
from ...holistic_landmarks import EYE_CONNECTIONS, HAND_CONNECTIONS, LIP_CONNECTIONS, POSE_CONNECTIONS, SELECTED_FACE_CONNECTIONS

POSE_EDGES = sorted(POSE_CONNECTIONS)
HAND_EDGES = sorted(HAND_CONNECTIONS)
SELECTED_FACE_INDICES = sorted({index for edge in SELECTED_FACE_CONNECTIONS for index in edge})
MOUTH_INDICES = sorted({index for edge in LIP_CONNECTIONS for index in edge})
MOUTH_POSITIONS = [SELECTED_FACE_INDICES.index(index) for index in MOUTH_INDICES]
MOUTH_EDGES = [(MOUTH_INDICES.index(start), MOUTH_INDICES.index(end)) for start, end in LIP_CONNECTIONS]
FEATURE_CONNECTIONS = sorted(set(LIP_CONNECTIONS) | set(EYE_CONNECTIONS) | {(10, 1), (1, 152)})
FEATURE_INDICES = sorted({index for edge in FEATURE_CONNECTIONS for index in edge})
FEATURE_POSITIONS = [SELECTED_FACE_INDICES.index(index) for index in FEATURE_INDICES]
FEATURE_EDGES = [(FEATURE_INDICES.index(start), FEATURE_INDICES.index(end)) for start, end in FEATURE_CONNECTIONS]


def _numeric_csv(source: Path, output: Path) -> None:
    frame = pd.read_parquet(source)
    for column in ("head_direction", "gaze_direction"):
        if column in frame:
            frame[column] = frame[column].astype("category").cat.codes.replace(-1, np.nan)
    metadata = {"clip_id", "window_id", "track_id", "split", "label", "class_code", "window_phase", "binary_label"}
    feature_columns = [c for c in frame.columns if c not in metadata]
    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame.to_csv(output, index=False)
    schema = pd.DataFrame({"column": frame.columns, "is_feature": [c in feature_columns for c in frame.columns]})
    schema.to_json(output.with_name("feature_schema.json"), orient="records", indent=2)


def _draw_edges(axis, points, edges, color):
    if points is None or points.ndim != 2:
        return
    for start, end in edges:
        if end < len(points) and np.isfinite(points[[start, end], :2]).all():
            axis.plot(points[[start, end], 0], -points[[start, end], 1], color=color, linewidth=1.4)
    valid = np.isfinite(points[:, :2]).all(axis=1)
    axis.scatter(points[valid, 0], -points[valid, 1], s=8, color=color)


def _draw_track(image, data, frame_idx, color, label):
    def draw_frame_points(key, edges, color):
        points = data[key][frame_idx]
        if points is None or points.ndim != 2:
            return
        valid = np.isfinite(points[:, :2]).all(axis=1)
        pixels = np.zeros((len(points), 2), dtype=np.int32)
        pixels[valid] = np.clip(points[valid, :2], [0, 0], [image.shape[1] - 1, image.shape[0] - 1]).astype(np.int32)
        outline = (0, 0, 0) if color == (255, 255, 255) else None
        for start, end in edges:
            if start < len(points) and end < len(points) and valid[start] and valid[end]:
                if outline is not None:
                    cv2.line(image, tuple(pixels[start]), tuple(pixels[end]), outline, 4)
                cv2.line(image, tuple(pixels[start]), tuple(pixels[end]), color, 2)
        for point in pixels[valid]:
            if outline is not None:
                cv2.circle(image, tuple(point), 5, outline, -1)
            cv2.circle(image, tuple(point), 3, color, -1)

    draw_frame_points("pose_frame_lm", POSE_EDGES, color)
    draw_frame_points("left_hand_frame_lm", HAND_EDGES, color)
    draw_frame_points("right_hand_frame_lm", HAND_EDGES, color)
    face_points = data["face_frame_lm"][frame_idx]
    if face_points is not None and face_points.ndim == 2:
        feature_points = face_points[FEATURE_POSITIONS]
        valid = np.isfinite(feature_points[:, :2]).all(axis=1)
        pixels = np.zeros((len(feature_points), 2), dtype=np.int32)
        pixels[valid] = np.clip(feature_points[valid, :2], [0, 0], [image.shape[1] - 1, image.shape[0] - 1]).astype(np.int32)
        for start, end in FEATURE_EDGES:
            if valid[start] and valid[end]:
                cv2.line(image, tuple(pixels[start]), tuple(pixels[end]), color, 2)
        for point in pixels[valid]:
            cv2.circle(image, tuple(point), 3, color, -1)
    x1, y1, x2, y2 = (int(float(data[key][frame_idx])) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(image, (x1, max(0, y1 - 25)), (x1 + 240, y1), color, -1)
    cv2.putText(image, label, (x1 + 5, max(17, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def _select_front_tracks(tracks, frame_idx: int, max_n: int = 2, min_confidence: float = 0.5, min_area_ratio: float = 0.01):
    """Giữ tối đa 2 người rõ và gần camera nhất."""
    candidates = []
    for track in tracks:
        confidence = float(track["tracking_confidence"][frame_idx])
        width = float(track["bbox_x2"][frame_idx] - track["bbox_x1"][frame_idx])
        height = float(track["bbox_y2"][frame_idx] - track["bbox_y1"][frame_idx])
        frame_width = float(track["frame_w"][frame_idx])
        frame_height = float(track["frame_h"][frame_idx])
        area_ratio = width * height / max(frame_width * frame_height, 1.0)
        if confidence >= min_confidence and area_ratio >= min_area_ratio:
            candidates.append((width * height, track))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [track for _, track in candidates[:max_n]]


def _visualize_class(stage1_root: Path, raw_root: Path, windows: pd.DataFrame, class_code: str, output: Path, clip_prefix: str = "") -> None:
    selected = windows[windows["class_code"].astype(str) == class_code]
    if clip_prefix:
        selected = selected[selected["clip_id"].astype(str).str.startswith(clip_prefix)]
    if selected.empty:
        return
    frame_meta = pd.read_parquet(stage1_root / "selected_frames.parquet")
    tiles = []
    for _, row in selected.head(20).iterrows():
        candidates = sorted((raw_root / str(row["clip_id"]) / str(row["window_id"])).glob("track_*.npz"))
        if not candidates:
            continue
        tracks = [load_raw_npz(path) for path in candidates]
        local_count = min(len(track["pose_crop_lm"]) for track in tracks)
        frame_idx = max(0, local_count // 2)
        tracks = _select_front_tracks(tracks, frame_idx)
        frame_rows = frame_meta[frame_meta["window_id"].astype(str) == str(row["window_id"])].sort_values("window_local_index")
        if frame_rows.empty:
            continue
        frame_path = Path(str(frame_rows.iloc[min(frame_idx, len(frame_rows) - 1)]["frame_path"]))
        if not frame_path.is_absolute():
            frame_path = stage1_root / "frames" / frame_path
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        colors = [(40, 210, 40), (255, 255, 255)]
        for index, data in enumerate(tracks):
            track_id = int(data["track_id"][frame_idx])
            confidence = float(data["tracking_confidence"][frame_idx])
            _draw_track(image, data, frame_idx, colors[index % len(colors)], f"Track {track_id} | {confidence:.2f}")
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (255, 255, 255), -1)
        cv2.putText(image, f"class {class_code} | tracks {len(tracks)} | window {row['window_id']}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 0), 2)
        tiles.append(image)
    if not tiles:
        return
    tile_width = 420
    tile_height = int(tiles[0].shape[0] * tile_width / tiles[0].shape[1])
    tiles = [cv2.resize(tile, (tile_width, tile_height), interpolation=cv2.INTER_AREA) for tile in tiles]
    columns = min(4, len(tiles))
    rows = 1
    tiles = tiles[:columns]
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y, x = divmod(index, columns)
        sheet[y * tile_height:(y + 1) * tile_height, x * tile_width:(x + 1) * tile_width] = tile
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)


def _combine_class_sheets(visual_root: Path, output: Path) -> None:
    """Create overview with one row per class."""

    sheets = []
    for class_code in (f"c{index}" for index in range(1, 8)):
        candidates = sorted(visual_root.glob(f"*_{class_code}.png"))
        if not candidates:
            continue
        image = cv2.imread(str(candidates[0]))
        if image is None:
            continue
        image = cv2.resize(image, (840, 560), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (image.shape[1], 30), (255, 255, 255), -1)
        cv2.putText(image, class_code, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 120, 0), 2)
        sheets.append(image)
    if not sheets:
        return
    canvas = np.full((560 * len(sheets), 840, 3), 255, dtype=np.uint8)
    for index, image in enumerate(sheets[:7]):
        canvas[index * 560:(index + 1) * 560, :] = image
    cv2.imwrite(str(output), canvas)


def _sample_class_tiles(
    class_code: str,
    stage2_root: Path,
    stage1_root: Path,
    windows_by_split: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Lấy 2 train, 1 val, 1 test; thiếu split thì bù bằng train."""
    requests = ["train", "train", "val", "test"]
    tiles = []
    used: set[tuple[str, str]] = set()
    for requested_split in requests:
        candidates = windows_by_split.get(requested_split, pd.DataFrame())
        candidates = candidates[candidates["class_code"].astype(str) == class_code] if not candidates.empty else candidates
        candidates = candidates[~candidates.apply(lambda row: (requested_split, str(row["window_id"])) in used, axis=1)]
        actual_split = requested_split
        if candidates.empty:
            candidates = windows_by_split["train"]
            candidates = candidates[candidates["class_code"].astype(str) == class_code]
            candidates = candidates[~candidates.apply(lambda row: ("train", str(row["window_id"])) in used, axis=1)]
            actual_split = "train"
        if candidates.empty:
            continue
        row = candidates.iloc[int(rng.integers(len(candidates)))]
        used.add((actual_split, str(row["window_id"])))
        raw_dir = stage2_root / actual_split / "raw" / str(row["clip_id"]) / str(row["window_id"])
        paths = sorted(raw_dir.glob("track_*.npz"))
        if not paths:
            continue
        tracks = _select_front_tracks([load_raw_npz(path) for path in paths], 0)
        frame_meta = pd.read_parquet(stage1_root / actual_split / "selected_frames.parquet")
        frame_rows = frame_meta[frame_meta["window_id"].astype(str) == str(row["window_id"])].sort_values("window_local_index")
        if frame_rows.empty:
            continue
        frame_idx = min(len(frame_rows) // 2, len(frame_rows) - 1)
        frame_path = Path(str(frame_rows.iloc[frame_idx]["frame_path"]))
        if not frame_path.is_absolute():
            frame_path = stage1_root / actual_split / "frames" / frame_path
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        for index, data in enumerate(tracks):
            track_frame_idx = min(frame_idx, len(data["pose_frame_lm"]) - 1)
            _draw_track(image, data, track_frame_idx, [(40, 210, 40), (255, 255, 255)][index], f"Track {int(data['track_id'][track_frame_idx])}")
        cv2.putText(image, f"{actual_split} | {class_code}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0), 2)
        tiles.append(image)
    return tiles


def _create_7class_overview(stage2_root: Path, output: Path) -> None:
    rng = np.random.default_rng(20260803)
    splits = ("train", "val", "test")
    windows_by_split = {
        split: pd.read_parquet(stage2_root / split / "features" / "windows.parquet")
        for split in splits
    }
    rows = []
    for class_code in (f"c{index}" for index in range(1, 8)):
        tiles = _sample_class_tiles(class_code, stage2_root, Path("data/processed"), windows_by_split, rng)
        if not tiles:
            continue
        tile_width = 420
        tile_height = int(tiles[0].shape[0] * tile_width / tiles[0].shape[1])
        tiles = [cv2.resize(tile, (tile_width, tile_height), interpolation=cv2.INTER_AREA) for tile in tiles]
        while len(tiles) < 4:
            tiles.append(np.full_like(tiles[0], 255))
        rows.append(np.concatenate(tiles[:4], axis=1))
    if rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), np.concatenate(rows, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage 2 CSV and skeleton visualizations")
    parser.add_argument("--stage2-root", type=Path, default=Path("data/processed/stage2_landmarks"))
    parser.add_argument("--csv-root", type=Path, default=Path("data/processed/xgboost_csv"))
    parser.add_argument("--visual-root", type=Path, default=Path("data/processed/skeleton_visualizations"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--clip-prefix", default="", help="Only visualize clips starting with this prefix")
    parser.add_argument("--class-code", default="", help="Only visualize one class, e.g. c1")
    args = parser.parse_args()
    args.csv_root.mkdir(parents=True, exist_ok=True)
    args.visual_root.mkdir(parents=True, exist_ok=True)
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        split_root = args.stage2_root / split
        frames_path = split_root / "features" / "frames.parquet"
        windows_path = split_root / "features" / "windows.parquet"
        if not frames_path.exists() or not windows_path.exists():
            raise FileNotFoundError(f"Thiếu Stage 2 artifact cho {split}: {split_root}")
        _numeric_csv(frames_path, args.csv_root / f"{split}.csv")
        windows = pd.read_parquet(windows_path)
        for class_code in sorted(windows["class_code"].dropna().astype(str).unique()):
            if args.class_code and class_code != args.class_code:
                continue
            _visualize_class(Path("data/processed") / split, split_root / "raw", windows, class_code, args.visual_root / f"{split}_{class_code}.png", args.clip_prefix)
    if args.split == "all" and not args.class_code:
        _create_7class_overview(args.stage2_root, args.visual_root / "7class_2train_1val_1test.png")
    print(f"OK: CSV={args.csv_root}; skeletons={args.visual_root}")


if __name__ == "__main__":
    main()