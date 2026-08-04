import unittest
import tempfile
from pathlib import Path
import numpy as np

from backend.ai_services.pose_gaze.dataset.stage2.schemas import CropMeta, HolisticRaw
from backend.ai_services.pose_gaze.dataset.stage2.artifact import save_raw_npz, load_raw_npz

class ArtifactTests(unittest.TestCase):
    def test_save_and_load_npz(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            window_dir = Path(tmp_dir)
            meta = CropMeta(
                clip_id="clip_001",
                window_id="w000",
                frame_idx=0,
                track_id=1,
                bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10,
                crop_x1=0, crop_y1=0, crop_x2=12, crop_y2=12,
                frame_h=100, frame_w=100,
                timestamp_ms=0.0,
                tracking_confidence=0.9
            )
            raw = HolisticRaw(meta=meta)
            raw.pose_crop_lm = np.zeros((33, 3), dtype=np.float32)

            out_path = save_raw_npz(window_dir, 1, [raw])
            self.assertTrue(out_path.exists())

            loaded = load_raw_npz(out_path)
            self.assertIn("pose_crop_lm", loaded)
            self.assertEqual(loaded["pose_crop_lm"].shape, (1, 33, 3))
            self.assertEqual(loaded["track_id"][0], 1.0)

if __name__ == "__main__":
    unittest.main()
