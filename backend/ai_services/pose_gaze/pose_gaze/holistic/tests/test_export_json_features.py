from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from pose_gaze.holistic.feature_csv.export_json_features import export


class ExportProtocolTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow((
                "filename", "class_code", "action_start_s", "action_end_s", "split",
                "split_group", "camera_view_id", "media_path", "actor_ids",
                "action_actor_ids", "interaction_pairs",
            ))
            writer.writerow(("c1.mp4", "c1", "1", "2", "train", "g1", "front", "c1.mp4", '["s1"]', '["s1"]', "[]"))
            writer.writerow(("c5.mp4", "c5", "", "", "train", "g1", "front", "c5.mp4", '["s1"]', "[]", "[]"))
        json_dir = root / "json"
        json_dir.mkdir()
        payload = {
            "session_id": "session",
            "frames": [
                {"frame_id": index, "timestamp_ms": index * 1000, "tracks": [{"track_id": "1", "bbox_xyxy": [10, 10, 100, 200]}]}
                for index in range(4)
            ],
        }
        for name in ("c1", "c5"):
            (json_dir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
        return manifest, json_dir

    def _rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_all_excludes_background_and_keeps_non_target_as_c5(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, json_dir = self._fixture(Path(temporary))
            output = Path(temporary) / "all"
            export(manifest, json_dir, output, "2d_world", "all")
            rows = self._rows(output / "features_2d_world.csv")
            c1 = [row for row in rows if row["source_filename"] == "c1.mp4"]
            self.assertEqual(len(c1), 2)
            self.assertEqual(len([row for row in rows if row["class_code"] == "background"]), 0)
            self.assertEqual({row["target_state"] for row in c1}, {"event"})
            self.assertEqual({row["label"] for row in c1}, {"c1"})
            self.assertTrue(all(row["protocol"] == "all" for row in rows))
            mapping = self._rows(output / "actor_track_mapping.csv")
            self.assertEqual(len(mapping), 2)
            self.assertEqual({row["actor_id"] for row in mapping}, {"s1"})
            self.assertEqual({row["mapping_status"] for row in mapping}, {"provisional_needs_identity_review"})

    def test_action_only_keeps_only_event_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, json_dir = self._fixture(Path(temporary))
            output = Path(temporary) / "oracle"
            export(manifest, json_dir, output, "2d_world", "action_only")
            rows = self._rows(output / "features_2d_world.csv")
            c1 = [row for row in rows if row["class_code"] == "c1"]
            c5 = [row for row in rows if row["class_code"] == "c5"]
            self.assertEqual(len(c1), 2)
            self.assertEqual({row["target_state"] for row in c1}, {"event"})
            self.assertEqual(len(c5), 4)
            self.assertEqual({row["target_state"] for row in c5}, {"normal"})
            self.assertTrue(all(row["protocol"] == "action_only" for row in rows))

    def test_actor_mapping_labels_target_and_peer_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow((
                    "filename", "class_code", "action_start_s", "action_end_s", "split",
                    "split_group", "camera_view_id", "actor_ids", "action_actor_ids",
                    "interaction_pairs",
                ))
                writer.writerow((
                    "c2.mp4", "c2", "1", "2", "test", "g1", "front",
                    '["s19", "s20"]', '["s20"]', '[{"source":"s20","peer":"s19"}]',
                ))
            json_dir = root / "json"
            json_dir.mkdir()
            payload = {
                "session_id": "session",
                "frames": [
                    {
                        "frame_id": index,
                        "timestamp_ms": index * 1000,
                        "tracks": [
                            {"track_id": "1", "bbox_xyxy": [10, 10, 100, 200]},
                            {"track_id": "2", "bbox_xyxy": [300, 10, 400, 200]},
                        ],
                    }
                    for index in range(3)
                ],
            }
            (json_dir / "c2.json").write_text(json.dumps(payload), encoding="utf-8")
            output = root / "output"
            export(manifest, json_dir, output, "2d_world", "all")
            rows = self._rows(output / "features_2d_world.csv")
            c2_rows = [row for row in rows if row["source_filename"] == "c2.mp4"]
            self.assertEqual({(row["track_id"], row["actor_id"], row["class_code"]) for row in c2_rows if row["target_state"] == "event"}, {("2", "s20", "c2")})
            self.assertEqual({(row["track_id"], row["actor_id"], row["class_code"]) for row in c2_rows if row["target_state"] == "normal"}, {("1", "s19", "c5")})
            mapping = self._rows(output / "actor_track_mapping.csv")
            self.assertEqual({(row["actor_id"], row["track_side"]) for row in mapping}, {("s19", "left"), ("s20", "right")})

    def test_interaction_sources_mark_both_actors_as_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow((
                    "filename", "class_code", "action_start_s", "action_end_s", "split",
                    "split_group", "camera_view_id", "actor_ids", "action_actor_ids",
                    "interaction_pairs",
                ))
                writer.writerow((
                    "c3.mp4", "c3", "0", "2", "test", "g1", "front",
                    '["s9", "s10"]', '[]',
                    '[{"source":"s9","peer":"s10"},{"source":"s10","peer":"s9"}]',
                ))
            json_dir = root / "json"
            json_dir.mkdir()
            payload = {
                "session_id": "session",
                "frames": [{
                    "frame_id": 0, "timestamp_ms": 1000,
                    "tracks": [
                        {"track_id": "1", "bbox_xyxy": [10, 10, 100, 200]},
                        {"track_id": "2", "bbox_xyxy": [300, 10, 400, 200]},
                    ],
                }],
            }
            (json_dir / "c3.json").write_text(json.dumps(payload), encoding="utf-8")
            output = root / "output"
            export(manifest, json_dir, output, "2d_world", "all")
            rows = self._rows(output / "features_2d_world.csv")
            self.assertEqual({row["actor_role"] for row in rows}, {"action_source"})
            self.assertEqual({row["class_code"] for row in rows}, {"c3"})


if __name__ == "__main__":
    unittest.main()
