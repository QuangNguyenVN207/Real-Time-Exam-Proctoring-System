import unittest
import numpy as np
from backend.ai_services.pose_gaze.tracking.schemas import BoundingBox
from backend.ai_services.pose_gaze.dataset.stage2.crop import crop_with_padding, landmark_crop_to_frame

class CropTests(unittest.TestCase):
    def test_crop_with_padding_clamp(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(10, 10, 50, 50)
        crop, actual_bbox = crop_with_padding(frame, bbox, padding=0.15)
        # width = 40, padding = 6 px -> 4..56
        self.assertEqual(actual_bbox.x1, 4.0)
        self.assertEqual(actual_bbox.y1, 4.0)
        self.assertEqual(actual_bbox.x2, 56.0)
        self.assertEqual(actual_bbox.y2, 56.0)
        self.assertEqual(crop.shape, (52, 52, 3))

    def test_landmark_crop_to_frame(self):
        crop_bbox = BoundingBox(10, 20, 50, 60) # w=40, h=40
        fx, fy = landmark_crop_to_frame(0.5, 0.5, crop_bbox, 100, 200)
        # px = 10 + 0.5*40 = 30 -> 30/100 = 0.3
        # py = 20 + 0.5*40 = 40 -> 40/200 = 0.2
        self.assertAlmostEqual(fx, 0.3)
        self.assertAlmostEqual(fy, 0.2)

if __name__ == "__main__":
    unittest.main()
