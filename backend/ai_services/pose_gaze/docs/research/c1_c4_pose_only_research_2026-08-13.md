# C1/C4 pose-only research — 2026-08-13

## Scope and benchmark boundary

This is a read-only design audit. No production code, manifest, model, or benchmark artifact was changed.

The official reference remains the original-label causal pose-only C2/C3/C5 run at `tmp/behavior_actor_causal_pose_only_20260812/causal_actor_metrics.json`: actor macro-F1 `0.8730158730`, C2 F1 `1.0`, C3 F1 `0.7142857143`, C5 F1 `0.9047619048`, confusion `[[12,0,0],[0,5,3],[0,1,19]]`. The corrected-C3-manifest run is not an official benchmark and is not used below.

## Confirmed contract and current implementation gaps

- Truth and metrics must remain actor-level `(video, actor_id)`. `action_actor_ids` defines actor truth; `interaction_pairs` is relation metadata only. Full video is scanned, and live frame `t` may use only frame `t` plus state from earlier frames. Sources: [`BEHAVIOR_STAGE1_STAGE2_CONTEXT.md`](../../docs/BEHAVIOR_STAGE1_STAGE2_CONTEXT.md), [`CONTEXT.md`](../../pose_gaze/CONTEXT.md).
- The historical C1/C4 contract is a shared behavior family, not one motion: C1 and C4-v1 use deep head-down plus hand/arm departure toward the lower body; C4-v2 uses opposite-sleeve paired-limb proximity/overlap. Source: [`BEHAVIOR_STAGE1_STAGE2_CONTEXT.md`](../../docs/BEHAVIOR_STAGE1_STAGE2_CONTEXT.md), lines 83–190 and 1436–1459.
- Current [`c1_c4_pose_features.py`](../../pose_gaze/holistic/feature_csv/c1_c4_pose_features.py) loses the left/right hand identity by taking the maximum displacement across both arms and the maximum downward wrist delta. It therefore cannot represent “one hand writes while the other manipulates something near the pocket/lap.”
- Its baseline is the first 30 frames for every actor. This is unsafe when action starts near one second, and it cannot represent an actor whose hand is already on the lap at frame 0.
- Line 94 converts legitimate numeric zeroes to an empty string through `value or ""`; this conflates valid zero displacement with missing evidence.
- Lines 126–127 calculate the maximum run over the complete actor video and copy the final result onto every frame. That field is future-dependent and cannot enter a live-feed decision.
- The old branch has no honest C1-versus-C4-v1 separator. It only produces a shared C1/C4 episode flag. That limitation is semantic, not merely a missing threshold.

## Direct audit: `v_c1_s1_s2`

Sources: [`v_c1_s1_s2.MOV`](../../../../../data/raw_video/v_c1_s1_s2.MOV), canonical [`v_c1_s1_s2.json`](../../../../../data/raw_video/processed/holistic_outputs/v_c1_s1_s2.json), and [`holistic_manifest_front_v4.csv`](../../../../../data/raw_video/processed/holistic_manifest_front_v4.csv).

- Video has 303 frames at approximately 23.976 FPS and lasts 12.638 s. Manifest truth is C1 for both `s1` and `s2`, action interval 1–12 s, train `group_02`.
- Existing mapping data identifies `s1 -> track 2` (left actor) and `s2 -> track 1` (right actor).
- Visual review shows both actors manipulating a dark object near the lap/lower desk area. The object is often partly hidden by the desk or body. A pose-only feature may describe the movement but may not claim “phone.”
- Canonical frame 0 already contradicts a mandatory desk-rest seed. Define wrist vertical position as `(wrist_y - shoulder_mid_y) / (hip_mid_y - shoulder_mid_y)`, where `1.0` is the hip line. At frame 0, track 2 left wrist is `1.50`; track 1 left wrist is `1.01`. At least one wrist of each actor therefore begins at or below the hip/lap boundary.
- At frame 24, the same values remain `1.44` and `1.01`. A detector requiring `desk rest -> downward retraction` will miss these actors because the lower-body state precedes the observable action transition.
- The old shoulder-line head-depth scalar is not sufficient alone. For track 1 it changes from `-0.268` at frame 0 to `-0.165` at frame 192; for track 2, `-0.249` to `-0.153`, despite visibly deeper downward attention later. Shoulder rotation changes the reference line, so deep head-down needs a torso-relative angle/compression signal in addition to the existing baseline delta.

## Object detector boundary

- The detector deliberately does not confirm `cheat_sheet`; paper identity is delegated to tracking because the same sheet can flicker between paper labels. Source: [`object_detect.py`](../../../object_detect/object_detect.py), lines 250–254.
- The documented four-class checkpoint has no `test_paper`; ordinary exam paper may be labelled `cheat_sheet`. Current safe handling maps paper boxes to `paper_unknown`. Source: [`PAPER_TRACKING.md`](../../PAPER_TRACKING.md), lines 6–15 and 66–75.
- [`object_cues.py`](../../pose_gaze/object_cues.py) treats missing ownership as unknown, freezes baseline paper regions, and requires high-confidence non-baseline paper evidence. This is suitable as optional confirmation, not as the required C1/C4 behavior gate.
- Offline [`export_object_cues.py`](../../pose_gaze/holistic/feature_csv/export_object_cues.py) samples objects sparsely after baseline (`detect_interval_s=1.5`) and confirms paper after three detection updates. This cadence can miss short hand-to-pocket/lap transitions and is not equivalent to live per-frame object evidence.

## Recommended pose-only representation

All values below are actor-scale normalized, validity-aware, per hand, and causal.

1. **Body coordinate frame.** Use shoulder midpoint, hip midpoint, torso length, shoulder width, and same-side shoulder/hip. Keep image coordinates only after actor normalization.
2. **Per-hand state, never max-first.** Export separate left/right wrist and elbow features: position relative to hip line, same-side hip distance, torso-centre distance, elbow angle, wrist-to-elbow vector, speed, acceleration, path length, and bone-length consistency.
3. **Lower-body regions.** Use honest proxy names:
   - `wrist_lap_proxy`: wrist near/below hip line and inside the torso-width envelope;
   - `wrist_pocket_proxy`: wrist close to the same-side hip with a flexed elbow;
   - `wrist_below_desk_baseline_proxy`: wrist falls below its actor-specific stable upper-hand band.
   Pose alone cannot prove “under table,” because the table plane is not observed.
4. **Two causal transition paths.** Support both:
   - `upper_hand_rest -> lower_body_hold`, for actors beginning with a hand on the desk;
   - `baseline_lower_body -> lower_body_manipulation`, for cases like `v_c1_s1_s2`. The second path uses increased local path length/speed near hip/lap, repeated short excursions, and a return/hold pattern; it must not require a desk seed.
5. **Deep head-down support.** Combine the existing signed nose/eye/mouth shoulder-line deltas with head-to-torso angle, nose-to-shoulder-mid distance divided by torso length, vertical head compression, and torso pitch/shoulder translation. Head-down is supporting evidence only; normal writing is a mandatory C5 hard negative.
6. **Quality firewall.** Low wrist visibility, implausible bone-length change, left/right swap, or discontinuous jump yields `unknown`, not motion. Pose wrist continuity should outrank hallucinated hand landmarks when the real hand is below the desk.
7. **Causal temporal summaries.** Maintain rolling elapsed duration, count, max, path length, and first-qualified frame from past/current samples only. One qualified current frame may flag the actor, but its feature vector may encode a completed causal transition. Never calculate a full-video run and backfill earlier frames.

## Honest C1/C4 decision hierarchy

```text
shared_lower_body_candidate =
    qualified hand-to-lower-body transition/manipulation
    AND deep-head-down support or strong lower-body manipulation
    AND quality valid

c4_v2 = shared candidate + opposite-sleeve paired-limb evidence
c1    = shared candidate + reliable phone confirmation
c4_v1 = shared candidate + reliable new-paper confirmation
otherwise = c1_c4_family_unknown
```

Without a reliable phone/paper cue, C1 and C4-v1 are observationally overlapping in 2D pose. Forcing a subtype will encourage the model to learn actor, clothing, side, or video background instead of behavior. Pose-only work should first maximize actor-level recall/precision for the shared C1/C4 family; only C4-v2 has a defensible pose-only subtype cue today.

## Proposed experiment order

1. Canonically audit all C1/C4 actors for initial hand state (`upper`, `lap`, `pocket`, `unknown`) and variant (`C1-like lower-body`, `C4-v1`, `C4-v2`). This is audit metadata, not model input.
2. Implement the per-hand causal features above in an isolated branch; do not modify the current C2/C3/C5 benchmark artifact.
3. Train/evaluate a shared `C1/C4-family vs C5` actor specialist first. Split and thresholds remain train-only; report actor precision/recall/F1, confusion, false actors, and first-flag latency.
4. Run ablations: head-only; per-hand lower-body-only; combined; combined plus C4-v2 sleeve geometry. Reject any model that wins only through absolute image coordinates or identity/source metadata.
5. Add object confirmation later as optional subtype evidence. Do not let missing object detections suppress a valid behavior-family flag.

## Blocking conclusion

The next implementation should not train a five-class C1/C2/C3/C4/C5 classifier immediately. First implement and validate a causal actor-level C1/C4-family specialist with two onset paths, especially the baseline-lap path shown by `v_c1_s1_s2`. C1 versus C4-v1 remains unidentifiable from pose alone; claiming otherwise would be label fitting rather than behavior recognition.
