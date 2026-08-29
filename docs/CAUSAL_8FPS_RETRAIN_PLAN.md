# Causal 8 FPS Retraining Plan

## Outcome

Produce one actor-level causal model trained and calibrated on landmarks extracted at a true 8 FPS processing cadence. Runtime and offline replay must share timestamp-based temporal semantics while keeping capture and tracking environment-specific.

## Execution scope and recovery

- Sole working tree: `C:\Real-Time-Exam-Proctoring-System`. Do not create another worktree for this plan.
- Active implementation branch: `retrain-8fps`.
- Branch base: `origin/develop` at `44f81319f94d1ced2139a7d45e45e229d36e7fbf`.
- Recovery branch: `codex/safety-pre-retrain-8fps-20260824`.
- Recovery commit: `9394b86053679d1986822b2ed8bf1c4ea7c728ec` (`docs: retrain on 8fps: preserve pre-retrain baseline`).
- `bugfix/wingery-causal-webcam-runtime` is reference/recovery history only; do not implement this plan on it.
- Keep all plan code and generated 8 FPS artifacts on `retrain-8fps`. Existing unrelated dirty and untracked files are outside this plan and must remain untouched.
- Do not merge, push, or open a pull request unless the user explicitly approves that action.

This plan has one experiment branch:

```text
source video
  -> timestamp sampler at 8 FPS
  -> tracking and landmark extraction on sampled frames only
  -> causal 8 FPS feature generation
  -> grouped training and calibration
  -> actor-level causal replay
  -> live 8 FPS validation
```

The existing 30 FPS JSON, derived feature CSVs, model weights, and fitted thresholds are reference artifacts. They are not training input for the 8 FPS model.

## Repository inputs and outputs

- Manifest: `data/raw_video/processed/holistic_manifest_front_v4.csv`
- Source-video root: `data/raw_video`
- New landmark output root: `data/raw_video/processed/holistic_output_8fps`

Resolve each manifest video under the declared source-video root. Create the new output root without replacing or deleting existing landmark directories. JSON landmarks extracted under this plan belong only in `holistic_output_8fps`.

## Preserved 30 FPS reference benchmark

Preserve this historical artifact unchanged:

```text
tmp/benchmark_face_mesh_restored_cuda_snapshot_verify_final_20260820
```

Its `causal_actor_metrics.json` records:

- protocol: `actor_only_causal_live_feed_rolling_replay_extended_suspicious`;
- primary unit: `(video, actor_id)`;
- labels: `[suspicious_activity, c2, c3, c5]`;
- actor macro-F1: `0.826448522100696`;
- metrics-file SHA256: `673CD6C4EDE61278B4B60F399D569A434FE54917BB69ADEC199C8F1A73166AD9`.

This is the preserved 30 FPS comparison artifact, not an 8 FPS model or a source of reusable thresholds. The 8 FPS run must report its result beside this reference with both provenance contracts visible; numerical comparison alone does not establish parity.

## Fixed decisions

- Target period is `125 ms`; sampling occurs before tracking and landmark extraction.
- 24/30 FPS sequences are synthetic fixtures. Actual sources: 37 clips at `23.976` FPS; 66 at `29.922–30.026` FPS.
- Use each video's actual PTS; never round to fixture FPS or assume dataset-wide FPS.
- For targets `0, 125, 250, ... ms`, select the earliest crossing frame, then advance `125 ms`. Never use an integer stride.
- Preserve original `source_frame_index` and timestamp; `sample_index` is separate, contiguous, and monotonic.
- Drop-tail: emit only on finite, increasing PTS crossing the next target; never force the final frame. Record the tail gap.
- Missing, duplicate, regressing, or non-finite PTS fails the clip with explicit terminal status. Never synthesize timestamps.
- Nominal-FPS fallback stays disabled until a real-container CFR experiment proves exact source-index and failure-semantic parity.
- `source_fps` is provenance and diagnostic metadata only; it never determines timestamps, crossings, or frame keys.
- Code provenance records Git `HEAD` and LF-canonical SHA256 over the declared extractor scope.
- Hash input: sorted relative path, content length, and LF-canonical content for `batch_dataset.py` and every Step 1 sampler module.
- Scoped hash includes uncommitted and newly added sampler code.
- Rolling horizon is trailing `3000 ms`, including the current observation.
- Warmup is `500 ms` from baseline start; prediction also requires minimum valid observations.
- Same-timestamp actors form one immutable snapshot before any actor state update.
- Temporal derivatives use elapsed seconds from `timestamp_ms`.
- Keep approved actor/pair grouped splits; no split changes or locked-test tuning during FPS conversion.
- Modify existing extractor, feature, replay, and runtime modules in place, only for 8 FPS timestamp requirements.
- Do not delete and recreate equivalent working code.
- Keep every change uncommitted until the user explicitly permits a commit.

## Offline and camera cadence policy

Dataset extraction has a fixed target schedule of 8 FPS. Source frames rejected by that schedule must not enter tracking, smoothing, or landmark extraction.

Live camera observation cadence is `3–10 FPS`; dataset extraction remains fixed at `8 FPS`:

- process each genuinely received observation once;
- retain its monotonic capture timestamp;
- never duplicate, interpolate, or replay a frame to manufacture 8 FPS;
- use timestamp horizons and per-second derivatives across `3–10 FPS`;
- expose actual observation gaps and coverage instead of treating missing frames as zero-valued evidence;
- reset or invalidate derivatives/state according to the declared maximum gap policy.

The fixed 8 FPS training cadence is the target operating rate, not a claim that every live one-second interval contains exactly eight observations.

## Step 1: Freeze the 8 FPS data contract

Define and test one sampled-frame record with at least:

```text
clip_id
sample_index
source_frame_index
timestamp_ms
source_fps
target_fps = 8
sampling_policy
frame_width
frame_height
tracks
```

`sampling_policy` is a versioned structure, not a free-text label:

```text
version = causal_8fps_pts_v1
target_period_ms = 125
selection = earliest_finite_pts_at_or_after_target
end_of_stream = drop_tail
invalid_pts = fail_clip
nominal_fps_fallback = disabled
```

Required invariants:

- `sample_index` is contiguous within a clip.
- `source_frame_index` is strictly increasing but may skip values.
- `timestamp_ms` is strictly increasing.
- each selected frame is the earliest source frame at or after the next 125 ms target time, subject to an explicitly documented end-of-stream rule;
- provenance records the source-video hash, extractor configuration, and code revision.

### 2026-08-24 Step 1 decision experiments

**EXPERIMENT — EOS.** Forced-final adds `68/103` frames (`66.019%`); drop-tail gap p50/p95/max is `41/100/101 ms`. Decision: drop-tail. Evidence: `tmp/causal_8fps_step1_eos_experiment_20260824.json` (`65812B9B472BE52C285FED3F732583F39E5ED8CA3A51641BDC6AB4DADAD48FBB`).

**EXPERIMENT — PTS.** Exact 24/30 FPS indices are `[0,3,6,9,12,15,18,21]` and `[0,4,8,12,15,19,23,27]`; invalid PTS fails. Decision: actual PTS, `125 ms`, no nominal-FPS fallback.

**EXPERIMENT — input.** `data/raw_video` resolves `103/103`; `49,412` frames produce `13,767` samples; invalid PTS count is `0`. Evidence: `tmp/causal_8fps_step1_pts_preflight_20260824.json` (`83E7C64B8E1DBFDBDBC353AA36A6068457AF03FC0B787E23735D6F19AE56B318`).

**EXPERIMENT — provenance.** LF hash costs p50/p95 `0.234/0.334 ms`; Git `HEAD` costs `54.675/83.461 ms`. Decision: record both. Evidence: `tmp/causal_8fps_step1_provenance_experiment_20260824.json` (`F934A8DE7924FCFD168731198FD08BCF997B64C3284C4B0CA0F16505ABFC76D4`).

### Step 1 execution status — PASS / Gate 1

Evidence: focused sampling tests `8/8`; batch regression `14/14`; `py_compile` PASS. Synthetic 24 FPS indices: `[0, 3, 6, 9, 12, 15, 18, 21]`. Synthetic 30 FPS indices: `[0, 4, 8, 12, 15, 19, 23, 27]`. Manifest preflight: `103/103` clips, `49,412` decoded frames, `13,767` sampled frames, invalid PTS count `0`. Artifact SHA256 values are fixed above. Implementation: `backend/ai_services/pose_gaze/pose_gaze/holistic/batch_dataset/sampling.py`, `backend/ai_services/pose_gaze/pose_gaze/holistic/batch_dataset/batch_dataset.py`, and `backend/ai_services/pose_gaze/pose_gaze/holistic/tests/test_batch_dataset/test_8fps_sampling.py`. Invalid PTS fails extraction with no partial output publication. Provenance validation passed.

## Step 2: Extract a clean 8 FPS landmark dataset

Improve the existing batch extraction path so the sampler decides whether a decoded source frame enters tracking. Run tracking and landmark extraction only for accepted frames. Read videos selected by `data/raw_video/processed/holistic_manifest_front_v4.csv` from `data/raw_video`, and write landmark JSON to `data/raw_video/processed/holistic_output_8fps`. Preserve all existing landmark and 30 FPS artifacts.

Resume/checkpoint state must identify both the last decoded source frame and last emitted sample timestamp. Resuming must produce byte-equivalent sampled frame keys to an uninterrupted run.

Record per clip:

- decoded source-frame count;
- sampled-frame count;
- observed sampling interval distribution;
- tracking/landmark valid counts per actor;
- extraction failures;
- provenance hashes.

Completion criterion: every manifest clip has a terminal extraction status, and every successful output satisfies the Step 1 contract without reading landmark values from the old 30 FPS JSON.

### 2026-08-25 Step 2 execution status — PASS

Stage 2 extraction PASS. Ba tập stem manifest/output/ledger khớp chính xác `103/103`; ledger có `102` completed và `1` provenance-matching skipped, `0` failed, `0` pending, `0` retryable. Tổng `49,412` frame decoded tạo `13,767` sample. PTS thực tế đều hữu hạn, tăng nghiêm ngặt và tuân thủ earliest-crossing `125 ms`; `103/103` source hash khớp. Không đọc `action_actor_ids`, action truth hoặc input JSON 30 FPS; không còn `.part`, `.tmp` hay `.tracking_state`. Batch tests `21/21`, tracking tests `13/13` và `py_compile` PASS.

Command sau khi activate venv trong PowerShell:

```powershell
$env:PYTHONPATH='backend\ai_services\pose_gaze'; python -m pose_gaze.holistic.batch_dataset 'data\raw_video' --manifest 'data\raw_video\processed\holistic_manifest_front_v4.csv' --model 'weights\yolov8n.pt' --holistic-model 'weights\mediapipe\holistic_landmarker.task' --json-output-dir 'data\raw_video\processed\holistic_output_8fps' --device 0 --fail-fast
```

Command PowerShell bên ngoài Codex, không cần activate venv:

```powershell
$env:PYTHONPATH='backend\ai_services\pose_gaze'; & '.venv\Scripts\python.exe' -m pose_gaze.holistic.batch_dataset 'data\raw_video' --manifest 'data\raw_video\processed\holistic_manifest_front_v4.csv' --model 'weights\yolov8n.pt' --holistic-model 'weights\mediapipe\holistic_landmarker.task' --json-output-dir 'data\raw_video\processed\holistic_output_8fps' --device 0 --fail-fast
```

## Step 3: Make temporal feature semantics timestamp-based

Keep frame-local geometry definitions, but regenerate their values from the new 8 FPS landmarks. Replace observation-dependent temporal formulas with elapsed-time formulas.

At minimum, inspect and convert:

- hand and fingertip speed;
- hand raise speed;
- pair convergence rate;
- hand direction rate;
- head/torso velocity;
- any streak, gap, age, onset, or baseline expressed in frames;
- every `observations[:N]`, `len(...) < N`, `max_frames`, `warmup_frames`, and frame-delta calculation on the selected feature path.

Let `p` be the previous valid observation:

```text
dt_s = (timestamp_ms[t] - timestamp_ms[p]) / 1000.0
speed = norm(position[t] - position[p]) / dt_s
velocity = (value[t] - value[p]) / dt_s
max_derivative_gap_ms = 450
derivative_valid = measurement_valid[t] and measurement_valid[p] and continuity_valid and 0 < dt_s <= 0.450
continuity_valid = same_track and no_tracker_miss and not_reacquired and not_reset
```

Continuity is independent of elapsed gap. Invalid derivatives emit `mask=0`; never fabricate zero/spike. Fit numeric thresholds again after unit changes.

Completion criterion: the selected model feature list has an explicit classification of `frame_local`, `baseline_relative`, `rate_per_second`, or `window_aggregate`, and no selected temporal feature derives elapsed time from frame count.

### Stage 3 Gate 1 — Proposed Work Packet 1: C2/C3 rates

- Goal: implement the derivative contract for selected C2/C3 rate features.
- Oracle: formulas, `450 ms` gap, independent continuity mask above; focused `3/8/10 FPS`, miss/reacquisition/reset tests.
- Ownership: coder writes `behavior_subset_stage2.py`; tester writes `test_behavior_features.py` and `test_causal_stream.py`; both may read the selected feature path and focused tests.
- Exclude: plan, extractor, dataset regeneration, training, thresholds, Stage 4 window.
- Commands: focused `rg`; `py_compile`; named focused `unittest` modules.
- Stop: ambiguity, oracle failure, or required out-of-scope change.

## Step 4: Introduce one causal timestamp-window module

Place ordering, eviction, warmup, gap handling, and aggregation behind one stateful module interface. Callers provide the current timestamp and current frame snapshot; callers do not manage deques or convert milliseconds to frame counts.

Conceptual interface:

```python
window.update(
    frame_index=source_frame_index,
    timestamp_ms=timestamp_ms,
    features=current_features,
)
```

For an update at time `t`:

1. Reject a timestamp that is not strictly newer than the actor's previous timestamp.
2. Append the current observation.
3. Evict observations with `timestamp_ms < t - 3000`.
4. Aggregate only observations in `[t - 3000, t]`.
5. Return window start/end timestamps, duration, valid counts, coverage, and aggregate features.

Warmup is incremental. The actor becomes ready only after at least `500 ms` has elapsed from its baseline start and the configured minimum number of valid observations exists. Before readiness, emit `warmup_ready=0`; do not compute an earlier prediction using a baseline learned from later frames.

For peer features, build all actors' frame-local rows from one immutable timestamp snapshot, then update every actor window. Actor iteration order must not change output.

Completion criterion: prefix replay produces identical outputs for every timestamp shared by a short input and a longer input, and reversing actor iteration order produces identical same-timestamp pair features and decisions.

## Step 5: Regenerate the complete 8 FPS feature dataset

Run the normal feature stages from the clean 8 FPS landmark root. Do not downsample an existing aggregate CSV because rolling statistics, baselines, rates, validity, and tracker-derived values would retain 30 FPS semantics.

Write a feature manifest containing:

- input landmark-root hash;
- feature schema and ordered feature-name hash;
- temporal policy: `target_fps=8`, `warmup_ms=500`, `window_ms=3000`, derivative gap;
- row counts by clip and actor;
- validity/coverage distribution;
- split assignment hash.

Completion criterion: every training row resolves to a sampled 8 FPS landmark frame, all feature names and ordering are stable, and no input path points to a 30 FPS derived artifact.

## Step 6: Retrain and calibrate on grouped 8 FPS data

Train using the existing approved grouped split protocol. Perform model selection and threshold/calibration fitting only on training/grouped OOF data. Regenerate class weights and thresholds; do not copy the 30 FPS values.

Save one artifact bundle containing:

- model files and hashes;
- ordered feature schema;
- sampling and timestamp-window policy;
- training-data and split hashes;
- fitted thresholds/calibration;
- grouped OOF actor/pair predictions and metrics;
- reproducible command and environment metadata.

Completion criterion: the artifact loader rejects a mismatched feature schema or temporal policy, and the saved model can reproduce its recorded grouped OOF predictions from the saved 8 FPS feature input.

## Step 7: Validate offline replay and live runtime

Replay full videos causally at the same 8 FPS sampling policy. Then run a bounded live-camera validation at target 8 FPS. Keep camera capture/tracking telemetry separate from offline metrics.

Capture enough trace data to follow:

```text
timestamp
-> sampled/rejected
-> actor/peer validity
-> frame-local features
-> timestamp window
-> model scores
-> gates
-> current actor class
-> history event
```

Report actor-level class metrics, C5 false alerts, coverage, first-flag latency in milliseconds, observed processing FPS, and end-to-end latency. Do not infer camera quality from offline F1 alone.

Completion criterion: offline replay loads the exact promoted bundle, live runtime reports the same schema and temporal-policy hashes, and both paths satisfy the gates below.

## Optional robustness extension: duration-weighted aggregates

This extension is recorded for the observed `3–10 FPS` camera variation. It is not part of the official conversion steps and must not delay completion of the timestamp-causal 8 FPS retrain unless the focused runtime gate shows sampling-density drift that invalidates the result.

If activated as a separately approved improvement:

1. compute `mean`, `std`, and `q95` with duration weights;
2. retain ordinary `max` and `min`;
3. add `window_duration_ms`;
4. add `valid_observation_count`;
5. add `coverage_ratio`;
6. add `max_inter_observation_gap_ms`.

Activation requires one concrete failing assertion, such as materially different aggregate values for the same timestamped motion observed at `3 FPS` versus `10 FPS`. Keep the original unweighted timestamp-window implementation available as the official baseline for comparison; do not silently change the feature schema of an existing artifact.

## Focused gates

### Gate 1: True 8 FPS extraction

Pass when exact synthetic 24 FPS and 30 FPS fixtures match their declared source-frame indices, all 103 manifest clips are sampled from finite strictly increasing actual media PTS by 125 ms targets, rejected frames produce zero tracking and landmark calls, sampled timestamps are monotonic, and extraction provenance resolves through `holistic_manifest_front_v4.csv` to the source video and `holistic_output_8fps`.

This gate answers only: "Was the dataset actually extracted under the intended 8 FPS observation process?"

### Gate 2: Causal timestamp semantics

Pass when prefix invariance holds, rolling membership is exactly `[t - 3000, t]`, warmup never reads a later observation, derivatives use timestamp deltas, and actor iteration order does not affect same-timestamp peer results.

This gate answers only: "Can any feature or decision at `t` depend on data after `t` or on frame-rate-dependent counting?"

### Gate 3: Training provenance and actor-level validity

Pass when model selection/calibration uses grouped training/OOF data only, every training row originates from the 8 FPS landmark root, feature/policy hashes match the saved bundle, and primary metrics are actor/pair-level.

This gate answers only: "Was the promoted artifact trained and evaluated under the declared 8 FPS actor-level contract?"

### Gate 4: Runtime compatibility

Pass when offline replay and live runtime load the same promoted artifact, feature schema, temporal policy, and thresholds; the live run sustains its declared target or reports misses/gaps explicitly; and bounded neutral/action trials produce interpretable timestamped traces.

This gate answers only: "Can the trained contract execute in the intended live environment without hidden semantic drift?"

## Audit trigger and stopping rule

Run an audit only when a focused gate fails or current evidence contradicts the declared artifact/data contract. Scope the audit to the failed assertion and stop when one of these conditions is reached:

- the assertion passes after a fix;
- evidence shows the assertion was invalid and the plan is explicitly revised;
- progress requires a user decision that changes the objective.

A passing focused gate is final for the unchanged code and artifact hashes. New audits require new contradictory evidence, a relevant code/data change, or a user request.

## Promotion rule

Promote only after all four gates pass for the same code revision, data hashes, model hash, feature-schema hash, and temporal-policy hash. A failure returns work to the step that owns that assertion; it does not open a general repository audit.
