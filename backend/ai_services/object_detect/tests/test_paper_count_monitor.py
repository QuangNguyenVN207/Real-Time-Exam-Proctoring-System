from __future__ import annotations

import unittest

from backend.ai_services.object_detect.paper_count_monitor import (
    CountBasedPaperMonitor,
)


def _paper(bbox, confidence: float = 0.8, name: str = "paper_unknown"):
    return {
        "bbox_xyxy": list(bbox),
        "confidence": confidence,
        "class_name": name,
    }


BASELINE_PAPERS = [
    _paper([100, 500, 250, 650]),
    _paper([700, 500, 850, 650]),
]
NEW_LEFT_PAPER = _paper([350, 350, 450, 450], 0.91, "cheat_sheet")
PEOPLE = [
    {
        "track_id": 11,
        "person_id": "STUDENT_LEFT",
        "bbox_xyxy": [0, 0, 500, 800],
        "is_present": True,
    },
    {
        "track_id": 22,
        "person_id": "STUDENT_RIGHT",
        "bbox_xyxy": [500, 0, 1000, 800],
        "is_present": True,
    },
]


class CountBasedPaperMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.monitor = CountBasedPaperMonitor(confirmation_frames=3)
        self.monitor.create_session("exam")

    def _update(self, detections, frame_id: int):
        return self.monitor.update(
            "exam",
            paper_detections=detections,
            people=PEOPLE,
            frame_id=frame_id,
            timestamp_ms=frame_id * 100,
        )

    def _learn_and_arm(self) -> None:
        for frame_id in range(3):
            self._update(BASELINE_PAPERS, frame_id)
        self.monitor.arm("exam")

    def test_duplicate_and_partial_boxes_collapse_to_one_paper(self) -> None:
        clusters = self.monitor.cluster_papers(
            [
                _paper([100, 100, 300, 300], 0.70),
                _paper([120, 120, 280, 280], 0.90, "cheat_sheet"),
                _paper([600, 100, 800, 300], 0.80),
            ]
        )

        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0]["source_box_count"], 2)
        self.assertEqual(clusters[0]["confidence"], 0.90)

    def test_setup_uses_the_most_frequent_count(self) -> None:
        self._update(BASELINE_PAPERS, 1)
        self._update(BASELINE_PAPERS, 2)
        result = self._update(BASELINE_PAPERS + [NEW_LEFT_PAPER], 3)

        self.assertEqual(result["baseline_count"], 2)
        armed = self.monitor.arm("exam")
        self.assertEqual(armed["stable_count"], 2)

    def test_one_inference_spike_does_not_alert(self) -> None:
        self._learn_and_arm()

        spike = self._update(BASELINE_PAPERS + [NEW_LEFT_PAPER], 10)
        recovered = self._update(BASELINE_PAPERS, 11)

        self.assertEqual(spike["candidate_streak"], 1)
        self.assertEqual(spike["active_alerts"], [])
        self.assertEqual(recovered["stable_count"], 2)
        self.assertEqual(recovered["new_events"], [])

    def test_persistent_increase_alerts_nearest_assigned_person(self) -> None:
        self._learn_and_arm()
        detections = BASELINE_PAPERS + [NEW_LEFT_PAPER]

        self._update(detections, 10)
        self._update(detections, 11)
        result = self._update(detections, 12)

        self.assertEqual(result["stable_count"], 3)
        self.assertEqual(len(result["new_events"]), 1)
        event = result["new_events"][0]
        self.assertEqual(event["type"], "paper_count_increased")
        self.assertEqual(event["previous_count"], 2)
        self.assertEqual(event["current_count"], 3)
        self.assertEqual(event["owner_track_id"], 11)
        self.assertEqual(event["owner_person_id"], "STUDENT_LEFT")
        self.assertNotIn("paper_id", event)
        suspicious = [
            paper
            for paper in result["papers"]
            if paper["status"] == "suspicious_new_paper"
        ]
        self.assertEqual(len(suspicious), 1)
        self.assertNotIn("paper_id", suspicious[0])

    def test_return_to_baseline_clears_active_alert(self) -> None:
        self._learn_and_arm()
        increased = BASELINE_PAPERS + [NEW_LEFT_PAPER]
        for frame_id in (10, 11, 12):
            self._update(increased, frame_id)

        self._update(BASELINE_PAPERS, 20)
        self._update(BASELINE_PAPERS, 21)
        result = self._update(BASELINE_PAPERS, 22)

        self.assertEqual(result["stable_count"], 2)
        self.assertEqual(result["active_alerts"], [])
        self.assertEqual(result["new_events"][0]["type"], "paper_count_decreased")

    def test_per_person_paper_increase_detected_even_if_global_count_equals_baseline(
        self,
    ) -> None:
        self._learn_and_arm()
        # Suppose STUDENT_LEFT's baseline paper is missing, but STUDENT_RIGHT has 2 papers (baseline + new cheat sheet)
        # Total global count = 2 (same as global baseline), but STUDENT_RIGHT has 2 papers (exceeding STUDENT_RIGHT's baseline of 1)
        right_base_paper = BASELINE_PAPERS[1]
        new_right_paper = _paper([850, 100, 950, 200], 0.88, "cheat_sheet")
        per_person_spike_detections = [right_base_paper, new_right_paper]

        for frame_id in range(10, 13):
            result = self._update(per_person_spike_detections, frame_id)

        suspicious = [
            paper
            for paper in result["papers"]
            if paper["status"] == "suspicious_new_paper"
        ]
        self.assertEqual(len(suspicious), 1)
        self.assertEqual(suspicious[0]["owner_person_id"], "STUDENT_RIGHT")


if __name__ == "__main__":
    unittest.main()
