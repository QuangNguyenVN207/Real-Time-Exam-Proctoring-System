# Causal suspicious-activity handoff

## Read this first

This is an **extended experimental profile**, not the official benchmark.
The protected base branch is `codex/development-benchmark-v3`. Do not reset,
switch, commit, push, or overwrite it without user approval.

The ultimate unit is one decision per `(video, actor_id)`. Never report or
optimize video-level metrics. All input streams are full-video and causal:
frame `t` may use only frames `<= t`.

## Canonical inputs

| Purpose | Path |
| --- | --- |
| Actor truth, split, pairs | `data/raw_video/processed/holistic_manifest_front_v4.csv` |
| Temporal actor-frame geometry | `data/processed/behavior_stage1_stage2_features_v2/temporal_geometry_features.csv` |
| Canonical landmark JSON root | `data/raw_video/processed/holistic_outputs` |
| Authoritative benchmark runner | `backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/behavior_subset_stage2.py` |
| Causal state machine | `backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/causal_stream.py` |

Truth is derived only from manifest `action_actor_ids`: listed actors receive
the manifest class; unlisted actors are `c5`. `interaction_pairs` is used only
for explicit C2 prediction propagation and never overrides truth.

## Official benchmark goc - locked

Artifact:

`tmp/behavior_actor_causal_pose_only_20260812/causal_actor_metrics.json`

Metric labels: `[c2, c3, c5]`; actor macro F1: `0.873015873015873`.

Confusion matrix order `[c2,c3,c5]`:

```text
[[12, 0, 0],
 [ 0, 5, 3],
 [ 0, 1,19]]
```

Do not overwrite this artifact or substitute an extended-profile score for it.
Before claiming it is unchanged, hash the artifact before and after work.

## Implemented extended profile

One opt-in path was added to the existing runner; no independent benchmark
runner remains:

```powershell
$env:PYTHONPATH='backend\ai_services\pose_gaze'
& '.venv\Scripts\python.exe' -m pose_gaze.holistic.feature_csv.behavior_subset_stage2 `
  --input data\processed\behavior_stage1_stage2_features_v2\temporal_geometry_features.csv `
  --manifest data\raw_video\processed\holistic_manifest_front_v4.csv `
  --json-root data\raw_video\processed\holistic_outputs `
  --exclude-c7 --causal-replay --c3-pose-only --extended-suspicious `
  --output-dir tmp\behavior_actor_extended_suspicious_shared_causal_20260813
```

The profile maps source `c1` and `c4` actors to public
`suspicious_activity`; it keeps `c2,c3,c5`. C7 is intentionally excluded in
this causal profile. It reuses the existing `causal_aggregate_rows`,
`fit_binary_actor_model` (XGBoost), train-only actor-max thresholds and
`CausalSpecialistState`. One shared prefix stream avoids multiple full prefix
copies in RAM.

### Rule contract

- C2: existing finger/hand, explicit-pair midpoint gate; C2 propagates only to
  declared pair endpoints.
- C3: low hand/finger motion, peer-directed side turn, and hard head-down
  exclusion. Looking down must not become C3.
- suspicious_activity: head-down, active hand/finger movement, pose wrist
  below hip, and hand on the actor's own side outside C2 midpoint margin.
- State begins `c5`; pair C2 has priority. Otherwise a later class can replace
  the actor label only with stronger causal evidence. Evidence frame is the
  strongest accepted frame; first flag is the first accepted frame.

No truth, class code, actor/track ID, split/group ID, action interval, or
future row is a model/gate feature.

## Latest extended result - do not promote

Output directory:

`tmp/behavior_actor_extended_suspicious_shared_causal_20260813`

Files:

- `causal_actor_metrics.json` — actor-level metric, thresholds and leakage audit.
- `causal_specialist_predictions.csv` — one row per `(video, actor_id)` with
  truth, final class, strongest evidence frame and first flag frame.
- `causal_{c2,c3,suspicious_activity}_specialist.ubj` plus matching
  `*_feature_names.json` — trained experimental specialist artifacts.

Metric labels `[suspicious_activity,c2,c3,c5]`, 60 test actors:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| suspicious_activity | .778 | .350 | .483 |
| c2 | .600 | 1.000 | .750 |
| c3 | 1.000 | .625 | .769 |
| c5 | .654 | .850 | .739 |

Macro F1: `.685280`. Leakage audit: raw actor ID overlap `[]`, clip overlap
`[]`, split-group overlap `[]`, future rows `0`.

This is not acceptable for promotion: 7 of 20 suspicious actors became C2 and
6 became C5. It must not be compared numerically as a replacement for the
official C2/C3/C5 benchmark because labels and task differ.

## Required next work

1. Audit the 7 suspicious-to-C2 actors in `causal_specialist_predictions.csv`.
   Verify per-video canonical JSON midpoint evidence; do not relax or alter the
   manifest to improve F1.
2. Audit the 6 suspicious-to-C5 actors: head-down, hand movement, below-hip and
   own-side evidence frame-by-frame in canonical JSON. Determine whether the
   failure is landmark coverage, baseline, or rule semantics.
3. Do not run another train until that audit identifies one feature/gate change.
   Make one change, add a focused test, run all Holistic tests, hash official
   artifact before/after, then run exactly one extended benchmark.
4. Preserve C2 contract: high model score alone cannot flag C2; explicit
   midpoint evidence and pair restriction remain required.
5. Preserve C3 contract: head-down blocks C3; C3 is not a generic movement or
   look-down class.

## Verification command

```powershell
$env:PYTHONPATH='backend\ai_services\pose_gaze'
& '.venv\Scripts\python.exe' -m unittest discover `
  -s backend/ai_services/pose_gaze/pose_gaze/holistic/tests `
  -t backend/ai_services/pose_gaze -p 'test_*.py' -v
```

Latest completed result: `103` tests passed.
