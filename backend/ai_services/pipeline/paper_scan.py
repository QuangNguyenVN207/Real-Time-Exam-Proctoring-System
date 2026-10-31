"""Băm (ARM) baseline giấy và đóng gói kết quả PaperCountPipeline thành alert.

Sao y app_webrtc.py:343-401, bỏ phần xử lý nút bấm (offline arm theo mốc thời gian).
"""
from __future__ import annotations

from typing import Any, Optional

__all__ = ["arm_baseline", "build_alert"]


def arm_baseline(paper_pipeline: Any, session_id: str) -> dict[str, Any]:
    """Cố định baseline số tờ giấy đang có trên bàn. Trả `{status_msg, baseline_count}`."""
    state = paper_pipeline.arm_paper_monitoring(session_id)
    baseline_count = state.get("baseline_count", 0) if isinstance(state, dict) else 0
    return {
        "status_msg": f"ARMED! Baseline: {baseline_count} giấy.",
        "baseline_count": baseline_count,
        "state": state,
    }


def build_alert(result_paper: Optional[dict[str, Any]], timestamp: float) -> Optional[dict[str, Any]]:
    """Chuyển kết quả 1 frame của pipeline giấy thành alert cho UI/log.

    Alert `smartphone_detected` lọt sang pipeline giấy là do model nhầm điện thoại với tờ giấy,
    nên bị vứt — vật cấm đã có `object_scan` lo.
    """
    if not result_paper:
        return None

    raw_alerts = result_paper.get("alerts", [])
    papers = result_paper.get("papers", [])
    has_alerts = bool(raw_alerts)

    alerts = raw_alerts
    if has_alerts:
        first = alerts[0]
        if isinstance(first, dict) and first.get("label") == "smartphone_detected":
            alerts = []

    return {
        "module": "paper_count",
        "status": "alert" if has_alerts else "info",
        "timestamp": timestamp,
        "alerts": alerts,
        "papers": papers,
        "paper_count_state": result_paper.get("paper_count_state"),
        "people": result_paper.get("people", []),
    }
