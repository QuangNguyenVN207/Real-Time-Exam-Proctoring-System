# C1 pose-only causal actor benchmark — 2026-08-13

## Scope

This is a separate C1-vs-C5 experiment. It does not replace the official
C2/C3/C5 benchmark. Object detections are excluded because temporary
`backend/ai_services/object_detect/weights/best.pt` produced zero phone
detections and unstable cheating-paper detections on the C1/C5 test replay.

## Protocol

- Primary unit: `(video, actor_id)`.
- Truth: original `holistic_manifest_front_v4.csv`; action actors receive C1,
  other actors in the selected C1/C5 videos receive C5.
- Input: pose only; no object feature, source identifier, truth field, action
  interval, interaction relation, or future frame is a model feature.
- Replay: frame `t` uses current/past actor state only, with a 30-frame warmup
  and a bounded 90-frame rolling state.
- Threshold: actor max-prefix threshold selected using leave-one-train-group-out
  out-of-fold predictions. The test set is not used for threshold selection.

## Result

Artifact: `tmp/behavior_actor_c1_causal_pose_only_oof_20260813`

| Metric | Result |
|---|---:|
| C1 precision | 0.8571 |
| C1 recall | 0.6000 |
| C1 F1 | 0.7059 |
| C5 F1 | 0.8276 |
| Actor macro-F1 | 0.7667 |

Confusion for `[c1,c5]`: `[[6,4],[1,12]]`.

The four C1 false negatives are retained model limitations. The one C5 false
positive is retained as the primary hard negative. Do not alter the manifest
or lower the threshold against test actors.

## Ablation

- Head-only: C1 F1 `0.6667`, but 8/13 C5 actors false-alerted.
- Hands-only: C1 F1 `0.5333`, one C5 false alert, but 6/10 C1 actors missed.
- Combined head plus per-hand lower-body geometry: C1 F1 `0.7059`, one C5
  false alert, and 6/10 C1 actors detected.

The combined result supports the intended mechanism: head-down alone resembles
normal writing, while hand geometry alone misses too many positive actors.

## Reproduction

```powershell
python -m backend.ai_services.pose_gaze.pose_gaze.holistic.feature_csv.benchmark_c1_causal `
  --input data/processed/behavior_stage1_stage2_features_v2/temporal_geometry_features.csv `
  --manifest data/raw_video/processed/holistic_manifest_front_v4.csv `
  --output-dir tmp/behavior_actor_c1_causal_pose_only_oof_20260813 `
  --feature-family combined
```
