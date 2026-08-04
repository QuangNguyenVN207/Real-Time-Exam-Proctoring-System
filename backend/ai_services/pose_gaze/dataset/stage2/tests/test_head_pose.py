import unittest
from backend.ai_services.pose_gaze.dataset.stage2.head_pose import classify_head_direction

class HeadPoseTests(unittest.TestCase):
    def test_classify_head_direction(self):
        self.assertEqual(classify_head_direction(0.0, 0.0), "forward")
        self.assertEqual(classify_head_direction(-20.0, 0.0), "left")
        self.assertEqual(classify_head_direction(20.0, 0.0), "right")
        self.assertEqual(classify_head_direction(0.0, 20.0), "down")

if __name__ == "__main__":
    unittest.main()
