from __future__ import annotations

from pathlib import Path

TARGET_FPS = 10.0
WINDOW_SECONDS = 3.0
WINDOW_FRAMES = 30
WINDOW_OVERLAP_FRAMES = 10
# Taxonomy confirmed by the dataset owner: c5 is the only non-cheating class.
NON_CHEATING_CLASS_CODES = frozenset({"c5"})

# Stage 1 keeps the complete classroom context.  Frames are resized only when
# they exceed this analysis canvas; the source video is never cropped or
# overwritten.
ANALYSIS_MAX_WIDTH = 1280
ANALYSIS_MAX_HEIGHT = 720

# Selected frames are for annotation/review only.  Model windows still contain
# every sampled frame so temporal continuity is preserved.
SELECTED_FRAMES_PER_WINDOW = 6

# Quality score weights.  The score is intentionally based on cheap image
# measurements; pose/face quality belongs to Stage 2.
QUALITY_WEIGHT_SHARPNESS = 0.30
QUALITY_WEIGHT_BRIGHTNESS = 0.20
QUALITY_WEIGHT_PERSON = 0.20
QUALITY_WEIGHT_MOTION = 0.15
QUALITY_WEIGHT_DIVERSITY = 0.15


def stage1_root(output_root: Path | None = None, source_path: Path | None = None) -> Path:
    if output_root is not None:
        return output_root
    if source_path is not None:
        return source_path.parent / "stage1"
    return Path.cwd() / "stage1"
