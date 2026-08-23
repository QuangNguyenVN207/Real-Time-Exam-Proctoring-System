import unittest
from backend.ai_services.pose_gaze.pose_gaze.object_cues import assign_owner_by_center


class OwnerTests(unittest.TestCase):
    def test_nearest_center_assigns_paper(self):
        people = [{"track_id": 1, "bbox_xyxy": [0, 0, 100, 100]}, {"track_id": 2, "bbox_xyxy": [100, 0, 200, 100]}]
        self.assertEqual(assign_owner_by_center([20, 40, 40, 60], people)["track_id"], 1)
        self.assertEqual(assign_owner_by_center([160, 40, 180, 60], people)["track_id"], 2)

    def test_tie_is_deterministic(self):
        people = [{"track_id": 2, "bbox_xyxy": [100, 0, 200, 100]}, {"track_id": 1, "bbox_xyxy": [0, 0, 100, 100]}]
        self.assertEqual(assign_owner_by_center([90, 40, 110, 60], people)["track_id"], 1)


if __name__ == "__main__": unittest.main()
