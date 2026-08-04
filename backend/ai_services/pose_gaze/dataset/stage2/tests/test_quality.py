import unittest
import numpy as np
from backend.ai_services.pose_gaze.dataset.stage2.quality import face_quality_score, iris_present, is_gaze_valid

class QualityTests(unittest.TestCase):
    def test_face_quality_score_none(self):
        self.assertEqual(face_quality_score(None, None, 100, 100), 0.0)

    def test_iris_present(self):
        lm_468 = np.zeros((468, 3))
        lm_478 = np.zeros((478, 3))
        self.assertFalse(iris_present(lm_468))
        self.assertTrue(iris_present(lm_478))

    def test_is_gaze_valid(self):
        self.assertFalse(is_gaze_valid(0.5, True))
        self.assertFalse(is_gaze_valid(0.8, False))
        self.assertTrue(is_gaze_valid(0.7, True))

if __name__ == "__main__":
    unittest.main()
