# Front-v4 actor-aware rebuild

## Contract

The model unit is a temporal window for every actor track, not a whole video.
The front-v4 manifest supplies:

- `actor_ids`: actors in the recording;
- `action_actor_ids`: actors who perform the manifest action;
- `interaction_pairs`: source actor and peer actor relations. Every
  `source` is an action actor; every actor that is only a `peer` (or is not a
  source) is labeled normal `c5`;
- `class_code`: the action class `c1` through `c7`.

Actor IDs are ordered by their numeric suffix (`s19 < s20`). The smaller
actor is mapped to the left-most stable track and the larger actor to the
right-most stable track using the median track bounding-box center. Raw
`track_id` is temporary and is never treated as the actor identity.

If a video has more raw tracks than manifest actors, the exporter keeps the
tracks with the highest frame coverage, maps those tracks spatially, and
records the remaining tracks as `excluded_extra_track_needs_review` in the
mapping audit. The source manifest is not changed.

## Derived artifacts

- Manifest: `data/raw_video/processed/holistic_manifest_front_v4_actor_aware_debug.csv`
- Mapping audit: `data/processed/holistic_features_front_v4_actor_aware_mapping_audit_spatial/actor_track_mapping.csv`
- Frame features: `data/processed/holistic_features_front_v4_actor_aware_mapping_audit_spatial/features_2d_world_behavior.csv`
- Temporal dataset: `data/processed/holistic_temporal_front_v4_actor_aware_30_20/windows_2d_world_behavior.npz`
- Model: `weights/final_actor_aware_2d_world_behavior_summary_class_group/baseline.ubj`
- Predictions: `weights/final_actor_aware_2d_world_behavior_summary_class_group/test_actor_track_predictions.json`

The rebuilt dataset excludes `background` entirely and preserves actor metadata
in the NPZ. Every mapped actor is evaluated: action sources receive `c1`-`c7`
inside the annotated interval, while peers/non-sources receive normal `c5`.

## Metrics

The primary result is target-event actor-track evaluation:

- target-event actor-track macro-F1: `0.3621`;
- target-event actor-track accuracy: `0.4533`;
- actor-track-state macro-F1: `0.4062`;
- 8-class window macro-F1 (`background` + `c1`–`c7`): `0.3253`.

The old `0.4917` video metric is retained only as
`test_legacy_video_video_macro_f1`; it averages predictions from multiple
actors in one video and is not the primary behavior metric.
