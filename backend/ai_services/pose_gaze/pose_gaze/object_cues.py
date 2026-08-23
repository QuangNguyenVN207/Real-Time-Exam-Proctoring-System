"""Offline, identity-safe object and behavior cues for Holistic fusion.

Detector execution stays in the existing object_detect/PoseGazePaperPipeline.
This module only joins its output to an existing Holistic track and derives
finite, validity-aware features.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

OUTSIDE_PAPER_CONFIDENCE = 0.85
PHONE_CLASS_NAMES = frozenset({"phone", "smartphone", "cell phone"})
CHEATING_PAPER_CLASS_NAMES = frozenset(
    {"cheating_paper", "cheat_sheet", "possible_cheat_sheet"}
)
BASELINE_PAPER_CLASS_NAMES = frozenset({"baseline_paper", "test_paper"})


def _box(value: Any) -> tuple[float, float, float, float] | None:
    try:
        b = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None
    return b if len(b) == 4 and b[2] > b[0] and b[3] > b[1] else None


def iou(a: Any, b: Any) -> float:
    aa, bb = _box(a), _box(b)
    if aa is None or bb is None:
        return 0.0
    x1, y1 = max(aa[0], bb[0]), max(aa[1], bb[1])
    x2, y2 = min(aa[2], bb[2]), min(aa[3], bb[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area = (aa[2]-aa[0])*(aa[3]-aa[1]) + (bb[2]-bb[0])*(bb[3]-bb[1]) - inter
    return inter / area if area > 0 else 0.0


def match_person(track: dict[str, Any], people: Iterable[dict[str, Any]], threshold: float = 0.05) -> dict[str, Any] | None:
    """Match by IoU; return None on tie/insufficient evidence."""
    tb = track.get("bbox_xyxy") or track.get("bbox")
    scored = sorted(((iou(tb, p.get("bbox_xyxy") or p.get("bbox")), p) for p in people), reverse=True, key=lambda x: x[0])
    if not scored or scored[0][0] < threshold or (len(scored) > 1 and scored[0][0] == scored[1][0]):
        return None
    return scored[0][1]


def assign_owner_by_center(object_bbox: Any, people: Iterable[dict[str, Any]], max_normalized_distance: float = 0.75) -> dict[str, Any] | None:
    """Assign small object to person using bbox centers, never globally."""
    ob = _box(object_bbox)
    if ob is None:
        return None
    oc = ((ob[0] + ob[2]) / 2.0, (ob[1] + ob[3]) / 2.0)
    candidates = []
    for person in people:
        pb = _box(person.get("bbox_xyxy") or person.get("bbox"))
        if pb is None:
            continue
        pc = ((pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0)
        scale = max(math.hypot(pb[2] - pb[0], pb[3] - pb[1]), 1.0)
        inside = pb[0] <= oc[0] <= pb[2] and pb[1] <= oc[1] <= pb[3]
        candidates.append((0 if inside else 1, _dist(oc, pc) / scale, int(person.get("track_id", 10**9)), person))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    if not candidates or candidates[0][1] > max_normalized_distance:
        return None
    return candidates[0][3]


def object_row(track: dict[str, Any], result: dict[str, Any] | None, papers: list[dict[str, Any]] | None = None, *, baseline_frame: bool = False, baseline_paper_ids: set[int] | None = None, baseline_paper_boxes: list[Any] | None = None) -> dict[str, Any]:
    """Create one track-owned row. Unknown stays zero and invalid."""
    result = result or {}
    object_result = result.get("object_result") or {}
    raw = result.get("raw_objects") or object_result.get("raw_objects", [])
    people = result.get("people", [])
    baseline_paper_ids = baseline_paper_ids or set()
    baseline_paper_boxes = baseline_paper_boxes or []
    owner = str(track.get("track_id", track.get("student_id", "")))
    # Pipeline must provide owner. Missing owner is unknown, never broadcast.
    track_box = track.get("bbox_xyxy") or track.get("bbox")
    def baseline_owner_match(item: dict[str, Any]) -> bool:
        """Mask frozen baseline by owner and center, even when IoU drifts."""
        box = _box(item.get("bbox_xyxy"))
        if box is None:
            return False
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        item_owner = item.get("owner_track_id")
        if item_owner is None:
            assigned = assign_owner_by_center(box, people)
            item_owner = assigned.get("track_id") if assigned else None
        for base in baseline_paper_boxes:
            if isinstance(base, dict):
                base_box, base_owner = base.get("bbox"), base.get("owner")
            else:
                base_box, base_owner = base, None
            bb = _box(base_box)
            if bb is None or (base_owner is not None and str(base_owner) != str(item_owner)):
                continue
            bw, bh = bb[2] - bb[0], bb[3] - bb[1]
            if bb[0] - 1.5*bw <= cx <= bb[2] + 1.5*bw and bb[1] - 1.5*bh <= cy <= bb[3] + 1.5*bh:
                return True
        return False

    owned = [
        x for x in raw
        if (x.get("owner_track_id") is not None and str(x.get("owner_track_id")) == owner)
        or (x.get("owner_track_id") is None and str((assign_owner_by_center(x.get("bbox_xyxy"), people) or {}).get("track_id")) == owner)
    ]
    # A semantic baseline_paper is always an allowed exam sheet. It must never
    # enter direct class assignment, even when it is not in the frozen baseline.
    baseline_owned = [
        item for item in owned if item.get("class_name") in BASELINE_PAPER_CLASS_NAMES
    ]
    # Frozen baseline paper: repeated detections are discarded permanently.
    owned = [
        x for x in owned
        if x.get("class_name") not in BASELINE_PAPER_CLASS_NAMES
        and not (x.get("class_name") in {"cheating_paper", "cheat_sheet", "book", "paper_unknown"}
                and (baseline_owner_match(x) or any(iou(x.get("bbox_xyxy"), b.get("bbox") if isinstance(b, dict) else b) >= 0.30 for b in baseline_paper_boxes)))
    ]
    if baseline_frame:
        owned = []
    phones = [x for x in owned if x.get("class_name") in PHONE_CLASS_NAMES]
    # Paper detector is permissive for baseline registration. After baseline,
    # only a high-confidence paper outside the frozen baseline can become c4.
    cheats = [
        x for x in owned
        if x.get("class_name") in CHEATING_PAPER_CLASS_NAMES
    ]
    authorized = [p for p in (papers or []) if p.get("owner_track_id") in {track.get("track_id"), track.get("student_id")} and p.get("authorized") is True]
    boxes = [x.get("bbox_xyxy") for x in owned if _box(x.get("bbox_xyxy"))]
    active_papers = [p for p in (papers or []) if float(p.get("confidence", 0.0)) >= OUTSIDE_PAPER_CONFIDENCE and p.get("paper_id") not in baseline_paper_ids and not baseline_owner_match(p) and not any(iou(p.get("bbox_xyxy"), b.get("bbox") if isinstance(b, dict) else b) >= 0.30 for b in baseline_paper_boxes)]
    phone_score = max((float(x.get("confidence", 0)) for x in phones), default=0.0)
    cheating_paper_score = max(
        (float(x.get("confidence", 0)) for x in cheats), default=0.0
    )
    # Detection semantics are authoritative by user contract. If both are in
    # the same actor/frame, stronger object evidence wins deterministically.
    direct_object_class = ""
    direct_object_score = 0.0
    if phone_score > 0.0 or cheating_paper_score > 0.0:
        direct_object_class, direct_object_score = max(
            (("c1", phone_score), ("c4", cheating_paper_score)),
            key=lambda item: (item[1], item[0] == "c1"),
        )
    return {
        "holistic_track_id": owner,
        "object_owner": owner if owned or baseline_owned or authorized else "unknown",
        "phone_conf": phone_score,
        "cheat_sheet_conf": cheating_paper_score,
        "baseline_paper_conf": max(
            (float(x.get("confidence", 0)) for x in baseline_owned), default=0.0
        ),
        "direct_object_class": direct_object_class,
        "direct_object_score": direct_object_score,
        "authorized_paper_count": len(baseline_paper_boxes),
        "new_paper_count": sum(1 for p in active_papers if p.get("owner_track_id") in {track.get("track_id"), track.get("student_id")} ),
        "paper_alert": int(not baseline_frame and any(a.get("label") == "possible_cheat_sheet" and float(a.get("confidence", 0.0)) >= OUTSIDE_PAPER_CONFIDENCE and a.get("owner_track_id") in {track.get("track_id"), track.get("student_id")} and a.get("paper_id") not in baseline_paper_ids and not any(iou(a.get("bbox_xyxy"), b.get("bbox") if isinstance(b, dict) else b) >= 0.30 for b in baseline_paper_boxes) for a in result.get("alerts", []))),
        "object_bbox": boxes[0] if boxes else [],
        "object_valid": int(bool(owned or baseline_owned or authorized)),
    }


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


def behavior_features(row: dict[str, Any], previous: dict[str, Any] | None = None, dt_s: float = 1/24) -> dict[str, float]:
    """Derive stable scalar cues from exported landmark coordinates."""
    def point(prefix: str, index: int) -> tuple[float, float] | None:
        try:
            x, y = float(row[f"{prefix}_{index:03d}_x"]), float(row[f"{prefix}_{index:03d}_y"])
            return (x, y) if math.isfinite(x+y) else None
        except (KeyError, TypeError, ValueError): return None
    mouth = point("face", 13), point("face", 14)
    nose, left_wrist, right_wrist = point("pose", 0), point("pose", 15), point("pose", 16)
    values = {"mouth_open": _dist(*mouth) if all(mouth) else 0.0, "mouth_open_valid": float(all(mouth)), "wrist_object_distance": 0.0, "wrist_object_valid": 0.0}
    if nose and left_wrist and right_wrist:
        values["wrist_object_distance"] = min(_dist(nose, left_wrist), _dist(nose, right_wrist)); values["wrist_object_valid"] = 1.0
    if previous and dt_s > 0:
        values["mouth_velocity"] = (values["mouth_open"] - float(previous.get("mouth_open", 0))) / dt_s
    else: values["mouth_velocity"] = 0.0
    return values


__all__ = ["iou", "match_person", "assign_owner_by_center", "object_row", "behavior_features"]
