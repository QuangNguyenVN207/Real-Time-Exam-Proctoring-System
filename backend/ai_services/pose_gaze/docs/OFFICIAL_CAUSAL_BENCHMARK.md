# Official actor-level causal benchmark

`benchmark gốc` means the original-label C2/C3/C5 causal pose-only replay. It
does not mean the corrected-manifest sensitivity run and it does not include
the unfinished C1/C4 family specialist.

## Locked result

- Artifact: `tmp/behavior_actor_causal_pose_only_20260812/causal_actor_metrics.json`
- Primary unit: `(video, actor_id)`
- Labels: `[c2, c3, c5]`
- Actor macro-F1: `0.873015873015873`
- C2 F1: `1.0`
- C3 precision / recall / F1: `0.8333333333333334 / 0.625 / 0.7142857142857143`
- C5 F1: `0.9047619047619048`
- Confusion matrix: `[[12,0,0],[0,5,3],[0,1,19]]`
- `future_frames_used_for_decision`: `false`

## Provenance lock

- Git HEAD at promotion: `41601cc40185e2d4df16d52a71a56691fb7fe16d`
- Metrics SHA256: `09EA3D9165897B6DC4D192AFDE477DCC78C1887B99608C4A679C12D594904E6B`
- Manifest SHA256: `A99DC3C28C6511CEA0BEDE456F5F7F6C413B1B4C1415DE83549DA193D67ED236`
- Runner SHA256: `48018D983D8498FA78AAF5AA2C037D8F27AF370E50F1F23BB7016C5B226068DA`

## Reproduction command

Run from repository root with the project Python environment:

```powershell
python -m backend.ai_services.pose_gaze.pose_gaze.holistic.feature_csv.behavior_subset_stage2 `
  --input data/processed/behavior_stage1_stage2_features_v2/temporal_geometry_features.csv `
  --manifest data/raw_video/processed/holistic_manifest_front_v4.csv `
  --output-dir tmp/behavior_actor_causal_pose_only_rerun `
  --exclude-c7 `
  --causal-replay `
  --c3-pose-only
```

Acceptance requires exact agreement with the locked metrics and confusion
matrix above. If a provenance hash differs, report the drift before comparing
numbers. Do not edit truth to force agreement.

## Non-official result

`tmp/behavior_actor_causal_pose_only_corrected_manifest_20260812` is a
sensitivity analysis only. Its macro-F1 of about `0.9293` must not be called
the official or original benchmark.
