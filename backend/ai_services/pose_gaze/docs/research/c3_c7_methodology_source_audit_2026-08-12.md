# C3/C2/C7 methodology source audit

Date: 2026-08-12  
Scope: research sidecar only. No production code or benchmark artifact was changed.

## Evidence boundary

The two papers do not define the project's current C3/C2/C7 skeleton formulas. They provide related behavior names, temporal chunking, pose/RGB processing, and evaluation protocols. The actor-aware distinctions below come from the historical repository behavior documents and are not claimed as findings from the papers.

`RESEARCH.md` also cites a separate `data-07-00122` PDF; that source was outside this audit and is not used below.

## Paper facts

### `Exploring_the_potential_of_skeleton_and_machine_le (5).pdf`

- Dataset collection used 12 participants, consent, and three camera views (left, frontal, right) at about 3 m. The initial collection had six ~15-minute videos; later segmentation produced 710 clips of about 4–6 s and 100–130 frames, split into 310 cheating and 400 non-cheating clips. Listed cheating behaviors include paper exchange and looking at other examinees' papers. (pp. 3–4)
- The method extracts multi-person 2D YOLO-Pose skeletons, tracks detections with IoU, and stores keypoints as `(x, y, conf)` rows in CSV with a binary cheating/non-cheating label. The paper states an IoU threshold of 0.5 for associating consecutive detections. (pp. 4, 6–7)
- The model input is described as 51 features per frame. LSTM input is a 3-D `(n_samples, time_step, 51)` tensor; traditional ML input is flattened to `(n_samples*time_step, 51)`. The Ac-LSTM configuration is 10×51 input, four 50-unit LSTM layers, and dropout 0.2. (pp. 6–8)
- The paper reports XGBoost as the best listed model in its experiment: accuracy 0.95, precision 0.95, recall 0.90, F1 0.92. The reported data split is 80/20 over 10,277 cheating and 35,258 non-cheating rows. The cited evaluation section does not state a held-out-video, actor, group, or scene split. (pp. 8–9)
- The authors identify lower-body keypoints as often occluded and suggest removing them; they also report confusion between ordinary pondering/looking left or right and copying-answer behavior, and propose expanding the dataset. These are limitations/future work, not validated C3 features. (p. 10)

### `R4_CheatingVideoDescriptionBasedonSequencesofGestures_rev2 (2).pdf`

- The paper defines a gesture as an atomic action represented by a video chunk. A single-subject action is a gesture sequence; an interaction is a combination of two subjects' gesture sequences. Its target behaviors include giving codes, exchanging papers, and looking at a friend. (pp. 1–2)
- The dataset contains 71 cheating videos at 1280×720 with five classes: Give Sign/Code, Exchange Paper, Looking Friend, Talking, and CheatSheet; the paper says each cheating instance occurs once per video. (p. 3)
- This is an RGB/video-chunk method, not a skeleton-feature method: MOG foreground extraction creates 80×80 crops from 160×160 detections, grayscale video is chunked into 80×80×30 inputs, and a 3D-CNN produces a 512-D vector for XGBoost classification. (pp. 3–4)
- Temporal settings are 30-frame chunks with 10-frame overlap, giving a 20-frame prediction distance. The authors use this overlap to smooth gesture predictions and produce more frequent outputs for a nominal 30-fps real-time system. (p. 4)
- The paper reports an 80/20 split of gesture chunks after segmentation and reports accuracy/Kappa for gesture recognition (mean accuracy 0.811, mean Kappa 0.760). It does not report an actor-level metric, macro-F1, or a held-out-video/actor split in the cited setup. The NoCheat class is described as the hardest because its learned features are spread out. (p. 5)
- The paper's visualizations say some 3D-CNN activations appear to respond to motion/hand regions, but this is learned RGB representation evidence, not an explicit hand, head-orientation, midpoint, or gaze formula. (p. 5)

## Historical repo separation rules

### C3: gaze/head-orientation proxy

`BEHAVIOR_STAGE1_STAGE2_CONTEXT.md`, “c3 evidence” and “Proposed c3 definition”:

- C3 is a sustained source-actor head and/or upper-body turn toward the peer, with direction agreeing with peer position. It is explicitly a visual-orientation proxy, not true gaze or intent.
- Primary pose points are nose `0`; eyes `1,4` support; shoulders `11,12` define shoulder midpoint/scale; hips `23,24` define torso midpoint/scale. Optional mouth `9,10` is corroboration. Hand landmarks are not C3 primary signals in this historical contract.
- Core normalized features are `nose_lateral` and `torso_lateral`, converted to baseline-relative peer-directed deltas by the sign of peer position. Velocity is an onset cue only; final C3 requires displacement, optional torso support, and persistence.
- Temporal controls are `N_head`, `N_torso`, `N_overlap`, `G_merge`, and descriptive `P_peak`. A moderate sustained turn may qualify; one extreme frame may not. The historical JSON audit records a progressive turn near frames 12–49 and a later moderate sustained candidate near frame 326.

### C2: hand/exchange geometry

`BEHAVIOR_STAGE1_STAGE2_CONTEXT.md`, “c2 common rule from variants 1 and 2”:

- C2 requires pair validity, peer-directed attention, directed hand exchange, temporal support, and valid pair geometry. Face turn alone is insufficient.
- Hand evidence priority is valid finger landmarks, then pose wrists `16/15`, elbows `14/13`, and shoulders `12/11` only as a last fallback.
- The fixed pair midpoint is established from the first jointly valid actor centers. `near_midpoint_pre_cross` uses a fixed 10% initial actor-center-gap margin; entering that margin is the primary exchange cue and does not require crossing.
- `N_face`, `N_inner`, `N_pair`, `L_head_to_hand`, and `G_merge` are temporal controls. Their thresholds must be learned from C2 variants and C5 hard negatives; fixed frame numbers or object visibility are not general rules.

### C7: own-side hand gesture, explicitly not exchange

`BEHAVIOR_STAGE1_STAGE2_CONTEXT.md`, “c7 hard case 1” and “c7 calculation formulas”:

- C7 uses an own-side hand/finger gesture, optionally supported by peer-directed head/face movement, while explicitly rejecting `near_midpoint_pre_cross` and sustained shared-zone convergence.
- Hand evidence priority is finger landmarks, then wrists, elbows, and shoulders as last fallback. Raise displacement, motion speed, and optionally baseline-relative finger shape change are normalized by shoulder width or actor scale.
- The C2 midpoint is fixed from the first paired-valid frame. `own_side_c7` must remain farther from the midpoint than the C2 margin; a gesture near the pair boundary is not called C7 merely because it has not crossed.
- The frame rule requires valid actor evidence, own-side gesture evidence, no midpoint/shared-zone exchange, and either head support or an explicitly measured occlusion override. `N_gesture` and `G_gesture` control persistence/gap merging.
- Missing face or hand observations are unknown/uncertain evidence, not C5 evidence. A gap must not switch the selected hand stream or zero-fill missing shape/position.

## Evaluation pitfalls to preserve

- The papers' row/chunk 80/20 protocols are not evidence for actor-level generalization. Adjacent frames or overlapping chunks can share source-video content; the papers do not state a held-out-video/actor/group protocol in the cited evaluation sections.
- The repo contract requires the primary unit `(video, actor_id)`. `action_actor_ids` define truth; `interaction_pairs` define relation and do not replace actor truth. Report actor-level precision, recall, F1, coverage, false alerts, and confusion—not video-level metrics.
- `action_start_s/action_end_s` are audit truth for onset/end. Model input and evaluation scan the full video; do not crop input to the annotated interval. Predicted runs are not ground-truth actor episodes.
- Historical repo windows are length 30 with stride 20, matching the R4 cadence only as a temporal-window reference. Persistence must be evaluated at frame level before aggregation; one positive frame may be enough for the actor decision under the current contract, while a detector episode may still use learned debounce for evidence quality.
- Baselines and thresholds must be actor/video-specific where required and learned from C5 false positives. A fixed first-frame or first-30-frame baseline can absorb an action or a hard negative.
- Quality failure is not a behavior label: missing face/hand data remains uncertainty, and fresh re-inference output must not replace canonical JSON for exact frame/landmark audits.
- `C7_Q7_EXPERIMENT_REVIEW_2026-08-12.md` records one actor-score experiment and explicitly rejects locked-test threshold tuning. It is a historical diagnostic, not a paper result and not a basis for inferring a new benchmark.

## Bottom line

Use the papers for: class-name context, multi-person tracking, skeleton-vs-RGB methodological contrast, 30-frame/10-overlap timing precedent, and evaluation cautions. Use the repo's historical C3/C2/C7 documents for the actual distinction: peer-directed signed head/torso orientation versus midpoint/shared-zone hand exchange versus own-side hand gesture. No paper-backed benchmark result for the project's actor-level C3/C2/C7 task was found.
