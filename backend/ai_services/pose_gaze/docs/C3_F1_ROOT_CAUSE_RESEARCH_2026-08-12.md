# C3 F1 root-cause research and repair proposal

Date: 2026-08-12  
Scope: actor-only C3 in the causal live-feed benchmark; no production model was promoted by this report.

## Conclusion first

C3 F1 is low for two different reasons, not one:

1. The causal benchmark has 7 false positives (`c5 -> c3`) and 3 false negatives (`c3 -> c5`). The resulting C3 precision is `0.4167`, recall `0.6250`, F1 `0.5000`.
2. The current C3 specialist does not implement the documented C3 feature contract faithfully. It mixes face mesh/PnP features into the primary specialist and uses a feature named `c3_shoulder_toward_peer` that is derived from normalized shoulder width, not peer-directed torso orientation.
3. The threshold is calibrated on prefix rows, while the live actor decision retains the strongest score over all observed prefixes. This is a unit mismatch: the threshold is optimized at frame/prefix level, but the acceptance decision is actor level.

Raising or lowering `0.7511805892` is not a real repair. The seven C5 false-positive evidence scores are approximately `0.7537` to `0.9780`; most are not borderline. The feature semantics and temporal qualification must be repaired first.

## Benchmark evidence

Artifact:

`tmp/behavior_actor_causal_rolling_exclude_c7_fixed2_20260812/causal_actor_metrics.json`

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| c2 | 1.0000 | 1.0000 | 1.0000 | 12 |
| c3 | 0.4167 | 0.6250 | 0.5000 | 8 |
| c5 | 0.8125 | 0.6500 | 0.7222 | 20 |
| macro |  |  | 0.7407 | 40 |

Confusion matrix, labels `[c2, c3, c5]`:

```text
[[12, 0, 0],
 [ 0, 5, 3],
 [ 0, 7,13]]
```

The C3 false negatives are concentrated in two test clips:

- `1786173707206_53534665179931303_9030090460038115400`: `s12`
- `1786173707214_53534665179931303_9030090460038115400`: `s11`, `s12`

Both manifests label `s11` and `s12` as C3 action actors. This clustering is a warning: verify the actor-level annotation and the actual peer-directed motion in those frames before treating all three as ordinary model misses.

The seven C5 false positives are:

- `1786174175453...`: `s17`, `s18`
- `VID20260804140955`: `s16`
- `WIN_20260802_11_59_41_Pro`: `s12`
- `WIN_20260802_12_01_36_Pro`: `s12`
- `WIN_20260802_12_03_21_Pro`: `s12`
- `WIN_20260802_12_03_45_Pro`: `s12`

Their strongest C3 evidence scores are `0.7922`, `0.8459`, `0.9261`, `0.9134`, `0.7537`, `0.9354`, and `0.9780`. This is a separation problem, not only a threshold problem.

The historical offline exclude-C7 artifact had actor macro-F1 `0.8998` and C3 F1 `0.7692`. It is not a live benchmark because it uses full-video evidence. It must remain a comparison only; it cannot validate a camera pipeline.

## Confirmed code-level causes

### 1. C3 implementation and C3 contract disagree

The project C3 definition requires a sustained peer-directed orientation proxy using pose points `{0, 1, 4, 11, 12, 23, 24}`:

- baseline-relative `nose_lateral` as primary head evidence;
- baseline-relative `torso_lateral` as upper-body support;
- sign from the peer's current position;
- velocity as onset evidence;
- persistence before a C3 candidate is qualified.

The current `derive_behavior_motion()` implementation instead creates:

```text
c3_head_toward_peer     = side * nose_eye_lateral
c3_shoulder_toward_peer = side * shoulder_dx
```

`shoulder_dx` is `(right_shoulder.x - left_shoulder.x) / shoulder_width`. In a frontal view it is approximately the normalized shoulder width, so `c3_shoulder_toward_peer` is close to a side/layout signal (`about -1` for one side and `+1` for the other), not a torso turn toward the peer. This is not valid C3 evidence.

The implementation also does not use the required hip points `23/24` for the C3 specialist. Therefore the model is asked to learn C3 from a proxy bundle that is materially different from the documented definition.

### 2. The primary specialist is over-bundled

`C3_FEATURES` currently contains pose roll/eye proxies, six-point SolvePnP head orientation, 14 face-mesh features, and quality masks. The historical C3 contract explicitly says hand landmarks and face mesh are not primary C3 signals. Face mesh may be a controlled ablation/support signal, but it should not be allowed to define the first C3 baseline.

This bundle makes it possible for the model to learn camera, face-quality, tracking, or ordinary head-turn patterns that correlate with the train labels. That is consistent with the high-scoring C5 false positives, but feature causality still requires an ablation to prove which family causes them.

### 3. Missing geometry is converted to zero

Several paths convert missing landmarks to numeric zero. In particular, missing C3 face observations set the face feature family to zero, and missing pose sub-measurements fall back to zero. A missing measurement is uncertainty, not a neutral geometric pose. A quality mask is added afterward, but this still gives the tree a learnable missingness pattern and can create artificial class cues.

The repair should keep valid-count/unknown state explicitly and avoid treating an absent point as a real zero displacement.

### 4. Calibration unit does not match decision unit

`fit_binary_threshold()` fits the C3 threshold over every warm prefix row in the training stream (`8,854` positive prefixes and `37,943` negative prefixes). The live replay then lets any observed prefix cross the threshold and retains the maximum score at actor level.

This gives one actor many correlated votes during calibration, then applies a max-over-time rule at inference. A transient C5 peak is enough to become a permanent actor C3 flag. Threshold calibration must instead be based on train actors (or cross-validated train actors), using the same qualified-evidence/max-prefix rule that deployment uses.

### 5. The live state has no C3 persistence gate

`CausalSpecialistState` accepts C3 when one rolling-prefix score crosses the threshold. The rolling statistics provide history, but there is no explicit learned `N_head`/`N_torso` or `N-of-W` qualification. That is weaker than the documented C3 definition of a sustained turn and is a likely contributor to false alerts. It is a hypothesis to validate against per-frame score runs, not a claim that all seven false positives are one-frame noise.

## What the PDFs support—and what they do not

### Exploring the potential of skeleton and machine learning...

Useful facts: pp. 3–4 describe multiple camera views and the effects of occlusion, illumination, scale, and scene conditions; pp. 6–8 use pose keypoints with confidence and temporal input; p. 10 reports confusion between ordinary pondering/left-right looking and copying behavior.

The reported XGBoost F1 is a row-level 80/20 result over extracted samples. The paper does not establish held-out actor/video/group generalization. It is not comparable to this project's actor-level C3 F1.

### R4: Cheating video description based on sequences of gestures

Useful facts: pp. 1–2 define interaction as a relation between two subjects; pp. 3–4 use temporal chunks; p. 4 uses 30-frame chunks with 10-frame overlap and a 20-frame prediction stride for a real-time cadence.

This supports explicit peer context and temporal qualification. It does not justify importing its RGB/3D-CNN metric into the pose C3 benchmark, and it does not report the required actor-level macro-F1.

Source audit: [c3_c7_methodology_source_audit_2026-08-12.md](research/c3_c7_methodology_source_audit_2026-08-12.md).

## Repair proposal

### Phase 0 — audit labels and evidence before retraining

Create a read-only error table for every C3 FN and C5 FP containing:

- `(video, actor_id)`, source frame index, first flag, strongest evidence frame;
- required pose-point validity for `{0,1,4,11,12,23,24}`;
- peer validity and peer direction at that frame;
- raw `nose_lateral`, `torso_lateral`, signed deltas, velocity, and persistence counts;
- face observed/predicted state, but not as a C3 label;
- whether the manifest actor assignment agrees with the visible behavior.

Manually review the three clustered FNs first. If either actor is not visibly peer-directed, the problem is truth/definition, not model capacity.

### Phase 1 — rebuild a pose-only C3 specialist

Start with a small, contract-faithful feature family:

```text
nose_lateral
torso_lateral
delta_head_peer
delta_torso_peer
head_velocity_peer
torso_velocity_peer
peer_direction_valid
required_pose_valid
peer_track_valid
```

Use peer direction from the explicit peer's current/baseline position, not actor ID or a fixed global side. Use hips to compute torso scale/torso lateral. Remove `c3_shoulder_toward_peer`, raw shoulder width, face mesh, and PnP from the first A/B experiment. Add face mesh/PnP only as separate ablations, and retain them only if they improve actor-level C3 without increasing C5 false alerts.

Do not zero-fill missing required points. Carry valid counts/unknown state through the rolling aggregator. A C3 decision may abstain while uncertain; uncertainty must not silently become a C5 geometric negative during evidence calculation.

### Phase 2 — causal temporal qualification

At each live frame, update only current/past state. Define a C3 candidate only after a train-calibrated persistence rule such as `N-of-W` head evidence plus optional torso support. The raw threshold, `N`, `W`, and allowed gap must be learned/validated from training actors and C5 hard negatives.

This does not use future frames. It adds a debounce/qualification rule before the actor-level strongest evidence is accepted. Report first-flag latency because a stricter gate can improve precision while delaying alerts or increasing false negatives.

### Phase 3 — calibrate at actor level

For each train actor, replay the causal stream and compute the exact deployment statistic: strongest qualified C3 evidence observed up to the current/final prefix. Fit the C3 threshold and persistence parameters on actor-level train calibration data, preferably with actor-grouped cross-validation inside the training partition. Never tune them on the locked test actors.

Keep the final rule causal:

```text
current/past frame
 -> pose/peer validity
 -> qualified C3 evidence run
 -> first flag
 -> retain strongest qualified evidence for this actor
```

No final-video max, future interval, action truth, or source/peer annotation may enter live inference.

### Phase 4 — hard-negative data

Add or relabel C5 hard negatives for ordinary head turns, pondering, camera/track jitter, face-mesh dropouts, and one-sided posture changes. The PDFs specifically warn about ordinary looking/pondering confusion; this must be represented in the project's actor-level C5 training data, not handled by an arbitrary threshold.

## Benchmark gates for any promotion

Run the same causal actor benchmark and report:

- C3 precision, recall, F1, and its 3x3 actor confusion matrix;
- macro-F1 over `[c2,c3,c5]`;
- C3/C5 false-alert counts;
- first-flag latency and target-actor unflagged count;
- `future_frames_used_for_decision: false`;
- exact train-only calibration rule and artifact path.

Minimum technical gate: C3 must improve over F1 `0.5000` without reducing actor macro-F1 below `0.7407`, unless a deliberate precision/latency tradeoff is approved. A C3 F1 target above `0.70` is reasonable, but it must be earned on the locked actor test set after the contract and label audit; it must not be obtained by test-threshold tuning.

## Grill decisions that must be explicit

1. Is C3 truth “peer-directed head/upper-body orientation proxy” or “true gaze/intent”? The current repository contract supports only the former.
2. In the two group-06 C3 videos, are both listed actors genuinely required to be C3 actors, or is the manifest over-labeling one actor? The benchmark currently says both.
3. What maximum first-flag latency is acceptable for live demo? A persistence gate can reduce C5 false alerts but will delay C3 flags.

Until these decisions and Phase 0 are complete, changing specialist priority or choosing another fixed threshold would be premature.

## A/B result: opt-in pose-only causal branch

Implemented as an isolated opt-in path with `--c3-pose-only`; the historical
causal artifact was not overwritten.

Artifact:

`tmp/behavior_actor_causal_pose_only_20260812/causal_actor_metrics.json`

| Metric | Historical causal | Pose-only causal |
|---|---:|---:|
| C3 precision | 0.4167 | 0.8333 |
| C3 recall | 0.6250 | 0.6250 |
| C3 F1 | 0.5000 | 0.7143 |
| C5 false positives | 7 | 1 |
| Actor macro-F1 | 0.7407 | 0.8730 |

The pose-only branch uses the explicit peer and pose points `{0,1,4,11,12,23,24}`.
It computes baseline-relative signed head and torso displacement, velocity, and
validity masks. It also calibrates the threshold from the train actor's
max-prefix statistic, matching the deployment decision unit. The benchmark
still reports `future_frames_used_for_decision: false`.

This is a strong precision/separation result, not a final promotion: all three
previous C3 false negatives remain. The next work is to inspect their action
truth/temporal definition and add a causal persistence/recall experiment on
training actors only. Do not lower the threshold on the locked test set.

## Follow-up label audit

The three false negatives were checked against the manifest and sampled source
frames. Both clips label both `s11` and `s12` as C3 actors, but the sampled
frames consistently show the left actor turning toward the peer while the
right actor remains oriented toward the paper. The pose audit also shows that
all required C3 landmarks are valid, so this is not a missing-pose failure.

The second clip's `s11` has a moderate head/torso peer-directed candidate and
should remain a hard positive for recall work. The two `s12` cases have weak
head evidence and are likely actor-level over-labels, subject to manual review
of the full intervals. Full details and frame links are in
`tmp/behavior_actor_causal_rolling_exclude_c7_fixed2_20260812/c3_false_negative_label_audit_20260812.md`.

Therefore persistence was not blindly added: it cannot repair likely wrong
actor truth. Ground-truth changes require a separately named manifest and an
explicit protocol decision; the locked test benchmark remains unchanged.

## Corrected-manifest sensitivity benchmark

Per approval, a separate manifest was created:

`tmp/holistic_manifest_front_v4_c3_corrected_s12_excluded_20260812.csv`

Only the two group-06 C3 rows changed `action_actor_ids` from `["s11","s12"]`
to `["s11"]`. The original manifest and locked benchmark artifact were not
modified. `interaction_pairs` were preserved.

The same pose-only causal code, split, threshold procedure, 30-frame warmup,
90-frame rolling window, and test videos were rerun:

`tmp/behavior_actor_causal_pose_only_corrected_manifest_20260812/causal_actor_metrics.json`

| Metric | Locked labels | Corrected-manifest sensitivity |
|---|---:|---:|
| C3 support | 8 | 6 |
| C3 precision | 0.8333 | 0.8333 |
| C3 recall | 0.6250 | 0.8333 |
| C3 F1 | 0.7143 | 0.8333 |
| C5 F1 | 0.9048 | 0.9545 |
| Actor macro-F1 | 0.8730 | 0.9293 |
| Target actor unflagged | 3 | 1 |

The corrected run leaves one C3 FN (`clip 2/s11`) and one C5 FP
(`1786174175539.../s18`). The improvement is attributable to changing the
test truth for two actors, not to a new model capability. The corrected result
must therefore be reported as a label-sensitivity analysis, not as the new
official benchmark, until the corrected manifest is formally adopted.

## Leakage audit of the corrected artifact

The corrected artifact was scanned independently for train/test clip, split
group, and `(clip, actor_id)` overlap; target/provenance fields in the explicit
C2/C3 schemas; future paired-frame references; and threshold calibration scope.
All checks passed. Every clip had `paired_valid_source_frame = 0`, so the fixed
C2 midpoint was not initialized from a future frame. The C3 pose-only schema
was exactly the eight contract features and its threshold was calibrated from
train actor max-prefix statistics only.

Full result: `tmp/behavior_actor_causal_pose_only_corrected_manifest_20260812/LEAKAGE_AUDIT_20260812.md`.

One boundary remains explicit: the benchmark supplies actor peer relations from
the manifest. That is permitted by the current explicit-pair contract, but it
does not measure automatic peer discovery from an unconfigured camera feed.

## Sources inspected

- `backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/behavior_subset_stage2.py`
- `backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/causal_stream.py`
- `backend/ai_services/pose_gaze/docs/BEHAVIOR_STAGE1_STAGE2_CONTEXT.md`
- `RESEARCH.md`
- `Exploring_the_potential_of_skeleton_and_machine_le (5).pdf`, cited pages above
- `R4_CheatingVideoDescriptionBasedonSequencesofGestures_rev2 (2).pdf`, cited pages above
