import unittest
from backend.ai_services.pose_gaze.pose_gaze.object_cues import iou, match_person, object_row


class ObjectCueTests(unittest.TestCase):
    def test_iou_and_identity_match(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)
        self.assertEqual(match_person({"bbox_xyxy": [0, 0, 10, 10]}, [{"track_id": 4, "bbox_xyxy": [0, 0, 10, 10]}])["track_id"], 4)

    def test_unknown_never_becomes_global_object(self):
        row = object_row({"track_id": 4}, {"raw_objects": [{"class_name": "smartphone", "confidence": .9}]})
        self.assertEqual(row["object_owner"], "unknown")
        self.assertEqual(row["phone_conf"], 0.0)
        self.assertEqual(row["object_valid"], 0)

    def test_best_pt_phone_directly_maps_to_c1_for_owned_actor(self):
        row = object_row(
            {"track_id": 4, "bbox_xyxy": [0, 0, 100, 200]},
            {
                "people": [{"track_id": 4, "bbox_xyxy": [0, 0, 100, 200]}],
                "raw_objects": [{
                    "class_name": "phone",
                    "confidence": 0.91,
                    "bbox_xyxy": [30, 80, 60, 130],
                    "owner_track_id": 4,
                }],
            },
        )
        self.assertEqual(row["direct_object_class"], "c1")
        self.assertAlmostEqual(row["direct_object_score"], 0.91)

    def test_best_pt_cheating_paper_directly_maps_to_c4(self):
        row = object_row(
            {"track_id": 4, "bbox_xyxy": [0, 0, 100, 200]},
            {
                "people": [{"track_id": 4, "bbox_xyxy": [0, 0, 100, 200]}],
                "raw_objects": [{
                    "class_name": "cheating_paper",
                    "confidence": 0.88,
                    "bbox_xyxy": [20, 90, 80, 140],
                    "owner_track_id": 4,
                }],
            },
        )
        self.assertEqual(row["direct_object_class"], "c4")
        self.assertAlmostEqual(row["direct_object_score"], 0.88)

    def test_baseline_paper_never_flags(self):
        row = object_row(
            {"track_id": 4, "bbox_xyxy": [0, 0, 100, 200]},
            {
                "people": [{"track_id": 4, "bbox_xyxy": [0, 0, 100, 200]}],
                "raw_objects": [{
                    "class_name": "baseline_paper",
                    "confidence": 0.99,
                    "bbox_xyxy": [10, 20, 90, 70],
                    "owner_track_id": 4,
                }],
            },
        )
        self.assertEqual(row["direct_object_class"], "")
        self.assertEqual(row["direct_object_score"], 0.0)
        self.assertEqual(row["paper_alert"], 0)


if __name__ == "__main__": unittest.main()
