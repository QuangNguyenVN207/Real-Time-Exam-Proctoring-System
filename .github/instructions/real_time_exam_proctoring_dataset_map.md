# Real-Time Exam Proctoring Dataset Map

- Workspace: `C:\Real-Time-Exam-Proctoring-System`
- Main code root: `backend/ai_services/pose_gaze`
- Main data root: `data/processed`
- Raw/annotation working root: `data/raw_video`

## Current data locations
- Primary Holistic manifest: `data/raw_video/processed/holistic_manifest.csv`
- Parquet manifest: `data/raw_video/processed/manifest.parquet`
- Source spreadsheet: `data/raw_video/dataset.xlsx`
- Stage 0 reviewed manifest: `data/raw_video/Dataset/stage0_review/stage0_manifest.csv`
- Stage 1/2 legacy artifacts: deleted; no longer valid paths.
- Holistic JSON outputs: `data/raw_video/processed/holistic_outputs/`
- Current derived frame artifacts: `data/raw_video/processed/frames.parquet`, `windows.parquet`, `splits.parquet`
- Current NPZ/training artifacts: `data/processed/holistic_temporal/`, plus `holistic_features/`, `holistic_features_action_only/`, `holistic_temporal_action_only/`, `holistic_temporal_stride8/`

## Important code
- `backend/ai_services/pose_gaze/dataset/manifest.py`: legacy schema helper; Stage 1/2 builders removed.
- `backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/export_json_features.py`: exports canonical Holistic JSON to feature CSV; supports `--action-scope all|action_only`, default `all`.
- `backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/build_temporal_dataset.py`: builds temporal NPZ and train-only scaler.
- `backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/train_baseline.py`: XGBoost baseline training/evaluation.
- `backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/cross_validate_groups.py`: leave-one-group-out evaluation.

## Status / outdated
- `dataset.xlsx` exists at `data/raw_video/dataset.xlsx`; earlier note saying no XLSX was wrong.
- Root-level `data/processed/stage1` and `data/processed/stage2_landmarks` do not exist in current workspace. Do not use those paths.
- Stage 1/Stage 2 Python packages are active dataset-building code, but old assumptions that their artifacts live directly under `data/processed/` are outdated.
- `data/raw_video/Dataset/stage1/stage1_report.csv` is report output, not Stage 1 source-of-truth manifest.
- `data/raw_video/processed/holistic_manifest.csv` is current input for canonical Holistic export/training.
- `action_start_s/action_end_s` exist in manifest. Historical full-frame datasets labeled frames outside intervals as action are contaminated; use separate action-only rebuild for experiments.
- Rebuilding temporal windows does not increase independent video/group count; it only reduces label noise/pseudo-replication. More independent recordings/groups require new data.
