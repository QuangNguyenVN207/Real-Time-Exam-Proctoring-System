# Behavior pipeline context: Stage 1 / Stage 2

Status: design context, not a claim that the final detector is implemented.
Explain in details, raise questions / concern everytime you missed my idea, ask me for reassurance
Always backup your claim with real data, real numbers and landmark behavior from the json file. Given the example video, check the landmark json of such video.

manifest: C:\Real-Time-Exam-Proctoring-System\data\raw_video\processed\holistic_manifest_front_v4.csv
holistic_output: C:\Real-Time-Exam-Proctoring-System\data\raw_video\processed\holistic_outputs
raw_video: C:\Real-Time-Exam-Proctoring-System\data\raw_video
pose_gaze/pose_gaze: C:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze

# Actor-only behavior contract (canonical)

- must classify behavior of every mapped actor, never behavior of a whole video.
- `action_actor_ids` is the complete set of actors performing the manifest `class_code`.
- Every actor in `action_actor_ids` receives the manifest `class_code`.
- Every actor not in `action_actor_ids` receives c5 for that action annotation.
- `interaction_pairs` describes source/peer relation; it does not override `action_actor_ids` truth.
- Example: c2 with `action_actor_ids=["s1","s2"]` means s1=c2 and s2=c2. c2 with `action_actor_ids=["s1"]` means s1=c2 and s2=c5.
- Background must not be a classifier class; the intended classifier has c1-c7 only.
- Actor-to-track mapping remains numeric actor order (smaller s-ID is left, larger s-ID is right) mapped to spatial track order.
- One qualified positive frame is sufficient to assign actor action class.
- No persistence requirement; positive evidence may be intermittent.
- Actor state starts temporary c5. Higher-scoring later positive may relabel actor; later negative never reverts it. c5 score must be learned experimentally.
- All metrics must be actor-level, keyed by `(video, actor_id)`. Video-level metrics are forbidden.

## Non-negotiable data contract

- `data/processed/holistic_features_front_v4_deployment_all_video/features_2d_world_behavior.csv` is the old feature schema associated with the historical ~0.49 video result. Its original 491 model features must remain unchanged. 
- Pose is an additional, separate branch. Never write pose values into face `yaw`, `pitch`, or `roll`; never fabricate face landmarks.
- `event` means the cheating/non-cheating temporal state. It is not a behavior class.
- Manifest `action_start_s` and `action_end_s` are ground-truth action onset/end.
  They are used for temporal truth and audit. Model input and evaluation scan
  the full video; never crop input to this interval.

## Why the first Stage 1 design failed

The initial hard gate was:

```text
p(event) < threshold -> c5
p(event) >= threshold -> Stage 2 subtype
```

This caused c5 collapse because every gate miss was forced to c5. It must not be promoted as the final architecture.

## Required two-stage architecture

### Stage 1: normal-envelope / transition detector

Learn the distribution of normal c5 pose behavior from c5 training data. Stage 1 confirms that a sustained transition has left the normal envelope; it must not directly turn every negative event prediction into c5.

Output states:

```text
normal / uncertain / behavior_candidate
```

The uncertain state must not be converted into c5 or a fabricated subtype.

### Stage 2: behavior subtype model

Train only on cheating subtype classes:

```text
c1, c2, c3, c4, c7
```

Run Stage 2 only after Stage 1 confirms a sustained behavior transition. c5 is emitted by the normal branch, not by `1 - p(event)`.

Use a soft fusion such as:

```text
subtype_score(class) = behavior_score * subtype_probability(class)
```

If the score is uncertain, hold the state or return `unknown`.





### Confirmed behavior: c1 and c4

c1/c4, c2, c3, and c7 now have concrete example evidence. Their thresholds
remain provisional and require c5 hard negatives. c6 remains excluded pending
confirmed semantic evidence.

The supplied c1 example, reported as similar to c4:
- the original video is 1786174198344_53534665179931303_9030090460038115400.mp4
- Frames `0–33`: normal. Face is normal; head is not deeply down; pose arms/hands are on the desk near the initial position.
- Frames `33–188`: action transition. The person takes and places a cheating aid; the head goes deeply down; an arm moves away from its initial desk position and down from the desk.
- c1/c4 are not one universal motion pattern. c4 has at least two variants:
  c4-v1 resembles c1 (`head_down` plus downward hand/arm departure), while
  c4-v2 keeps the hand near the body and extracts the cheating aid from the
  opposite sleeve. Stage 1 must branch by confirmed variant evidence; it must
  not force c4-v2 through the c4-v1 downward-arm rule.

### Confirmed c4-v2 audit: `v_c4_s7`

The user-confirmed canonical audit for `v_c4_s7` is:

```text
video: data/raw_video/v_c4_s7.MOV
canonical JSON: data/raw_video/processed/holistic_outputs/v_c4_s7.json
demo: backend/ai_services/pose_gaze/pose_gaze/holistic/test_media/outputs/v_c4_s7_demo.mp4
source actor: s7 -> track 1
peer actor: s8 -> track 2
track switch or occlusion: no
canonical frame count: 268, source_frame_index 0-267
safe normal baseline: source frames 0-36
c4-v2 episode onset: source frame 64
head_down interval: source frames 68-219
paired sleeve/limb cue: source frame 64 through end of video
episode end: no return to normal observed; end remains final video frame
```

The c4-v2 cue is hand/arm movement toward the opposite sleeve, represented by
paired proximity or overlap of these pose landmark pairs:

```text
(16,15)  wrist pair
(14,13)  elbow pair
(12,11)  shoulder pair
```

Landmark `0` (nose) supports the separate `head_down` signal. Individual
landmarks `16`, `14`, or `12` must not be called the sleeve cue. The observed
`v_c4_s7` evidence has head-down support, but c4-v2 subtype implementation must
not assume that every c4-v2 example has identical head timing until more
canonical examples are audited.

### Pose landmarks and features for class c1 and c4

Use pose landmarks, not hand landmarks:

```text
nose 0, eyes 1/4, mouth 9/10, shoulders 11/12, elbows 13/14, wrists 15/16
```

## Exact head-down formula

Use the JSON `pose_landmarks` normalized `x,y` coordinates consistently. Do not mix normalized coordinates with pixel coordinates. For frame `t`:

```text
SL = pose[11]                         # left shoulder
SR = pose[12]                         # right shoulder
N  = pose[0]                          # nose
I  = (pose[1] + pose[4]) / 2          # eye midpoint
M  = (pose[9] + pose[10]) / 2         # mouth midpoint
W  = distance(SL, SR)                 # shoulder scale
```

The shoulder line is the line through `SL` and `SR`. For any point `P`, interpolate the shoulder-line y value at `P.x`:

```text
shoulder_y(P.x) = SL.y + (P.x - SL.x) * (SR.y - SL.y) / (SR.x - SL.x)
d(P,t) = (P.y - shoulder_y(P.x)) / W
```

In image coordinates, `d(P,t) > 0` means the point is below the shoulder line. The three baseline-relative signals are:

```text
delta_nose_t  = d(N,t) - median_baseline(d(N))
delta_mouth_t = d(M,t) - median_baseline(d(M))
delta_eye_t   = d(I,t) - median_baseline(d(I))
```

The nose is the mandatory primary signal. Mouth and eye are corroborating signals because they can change rapidly at the onset of the movement:

```text
nose_down_t = delta_nose_t >= T_nose
face_support_t = median(delta_mouth_t, delta_eye_t) >= T_support
head_down_candidate_t = nose_down_t AND face_support_t
```

Do not use a one-frame velocity as the final flag. For transition timing, optionally record:

```text
velocity_p_t = delta_p_t - delta_p_(t-1)
```

but require the displacement condition to remain true for the learned persistence run.

Primary features:

- `head_down`: the formula above; nose-to-shoulder-line displacement is primary, mouth/eye support is secondary.
- `arm_departure`: c1/c4-v1 downward elbow/wrist displacement from the normal baseline, normalized by shoulder width or torso length.
- `c4_v2_sleeve_proximity`: opposite-sleeve movement supported by proximity or overlap of pose pairs `(16,15)`, `(14,13)`, or `(12,11)`, with actor-scale normalization.
- `arm_to_torso` and `wrist_to_shoulder` relations.
- pose validity/visibility as quality signals only.

The user stated that pose landmarks do not disappear. Therefore landmark disappearance is not a behavior feature. A low visibility/presence value is only a quality/uncertainty signal; never replace it with a proxy coordinate or face angle.

## JSON audit of the supplied example

Source JSON:

```text
backend/ai_services/pose_gaze/pose_gaze/holistic/test_media/outputs/final_1786174198344_53534665179931303_9030090460038115400.json
```

The file contains 477 frames (`source_frame_index` 0-476), and frame 33 is at 1100 ms while frame 188 is at 6267 ms, consistent with approximately 30 FPS. It contains two tracks, so the target track must be fixed before fitting thresholds.

Using the exact formula above and baseline frames 0-33:

```text
track 1 baseline median [nose, mouth, eye] = [-0.10827, -0.02397, -0.25791]
track 1 frame 188                 = [-0.14356, -0.06843, -0.28139]
track 1 delta frame 188            = [-0.03530, -0.04446, -0.02348]

track 2 baseline median [nose, mouth, eye] = [-0.00511,  0.05128, -0.12465]
track 2 frame 188                 = [-0.20892, -0.13436, -0.31634]
track 2 delta frame 188            = [-0.20381, -0.18564, -0.19169]
```

This is a blocking data concern: under the image-coordinate sign convention above, the supplied JSON does not show `track 1` nose moving below the shoulder line at frame 188; its delta is negative. `track 2` shows a much larger coordinated change in nose, mouth, and eye. The implementation must not silently choose a track or flip the sign. Confirm which track corresponds to the supplied screenshots before using this example to fit a c1/c4 threshold.

## c6 decision: exclude from training



## c2 common rule from variants 1 and 2

The two variants differ in appearance but share the same behavior structure:

```text
normal prefix
-> actor turns/attends toward the peer
-> one or both inner hands move into the shared exchange zone
-> hand proximity/convergence persists as one exchange episode
```

Variant 1 is visually stronger in bbox expansion, hand convergence, and
paper-reaching geometry. Variant 2 is visually stronger in the inner-hand
finger approach using point indices 12 and 11. These are alternative
observations of the same c2 event; neither is a universal requirement.

### Common c2 decision rule

Define the pair-level c2 candidate as:

```text
pair_has_two_valid_actors
AND face_or_head_turn_toward_peer
AND hand_exchange_signal
AND temporal_persistence
AND pair_geometry_valid
```

where:

```text
hand_exchange_signal =
    near_midpoint_pre_cross
    OR sustained inner_approach
    OR hand_convergence with shared-zone proximity
```

`near_midpoint_pre_cross` is the primary c2 hand flag: one valid finger or
fallback pose point enters the 10% margin immediately before the fixed pair
midpoint, without requiring it to cross. `sustained_inner_approach` handles
frames before that margin. `hand_convergence` handles an exchange where both
hands move close while the bboxes remain separate. At least one hand must show
directed movement toward the shared zone; generic hand motion on the desk is
insufficient.

### Common landmark priority and fixed midpoint

The priority order is strict:

```text
1. valid finger landmarks from the hand arrays
2. pose wrists: 16/15
3. pose elbows: 14/13
4. pose shoulders: 12/11 (last fallback only)
```

Pose `12/11` are shoulders and are never the primary c2 cue. Finger points
must be evaluated first. The later pose groups are fallbacks only when the
higher-priority group is missing or invalid; a shoulder point must not
override a valid finger, wrist, or elbow signal.

Let frame 0 mean `source_frame_index=0`. If one actor is invalid at that frame,
use the first frame where both actor bboxes are valid, record that frame as the
midpoint baseline, and never silently recalculate the midpoint later:

```text
C_left_0      = center x of the left actor bbox at midpoint baseline
C_right_0     = center x of the right actor bbox at midpoint baseline
pair_mid_x_0  = (C_left_0 + C_right_0) / 2
D_actor_0     = C_right_0 - C_left_0
M_mid         = 0.10 * D_actor_0
```

`pair_mid_x_0` is fixed for every later frame. `D_actor_0` is the initial
camera-image distance between actor centers. `M_mid` is the 10% camera-x
margin before the midpoint. For a left actor, a selected point is near before
crossing when `pair_mid_x_0 - M_mid <= x <= pair_mid_x_0`. For a right actor,
it is near before crossing when `pair_mid_x_0 <= x <= pair_mid_x_0 + M_mid`.
The frame-level flag is:

```text
near_midpoint_pre_cross_t = any selected point enters its side's 10% margin
```

The flag does not require a crossing and may fire on one frame. Stage 1 may
still apply temporal hysteresis before promoting the frame flag to a behavior
episode. If camera motion or track drift invalidates the fixed baseline,
output `uncertain`; do not recompute `pair_mid_x_0` per frame.

The event must be evaluated as a pair first, then assigned to actors using the
manifest. Every actor in `action_actor_ids` receives the c2 actor label. A
peer-only actor not in `action_actor_ids` receives c5, even if the pair event
is c2. Do not label only the actor whose hand crossed; that would violate the
actor-aware manifest contract.

### Common parameter definitions

```text
T_face_peer       minimum peer-directed face/head displacement or ROI exit
N_face            consecutive frames required for the face/head cue
T_inner_velocity  minimum inner-hand speed after actor-scale normalization
T_inner_gap       maximum normalized gap between inner hands in exchange zone
M_mid_ratio       fixed pre-midpoint margin ratio; currently 0.10
N_inner           consecutive frames required for approach/margin evidence
T_pair_proximity  maximum normalized hand or shared-zone distance
N_pair            consecutive frames required for pair proximity/convergence
L_head_to_hand    allowed time lag from head turn onset to hand cue onset
G_merge           maximum gap when joining fragmented cues into one episode
```

`T_face_peer` and `N_face` prevent a normal glance or detector noise from
starting c2. `T_inner_velocity` and `N_inner` distinguish a reach from a
static hand position. `T_inner_gap` and `T_pair_proximity` measure exchange
closeness; they are normalized by actor scale and must not use raw crop-local
coordinates. `M_mid_ratio=0.10` creates the pre-crossing margin; it is not a
crossing threshold. `N_pair` rejects a single-frame overlap. `L_head_to_hand` encodes
the expected turn-then-reach order but is a tolerance, not the manifest's
`action_start/action_end`. `G_merge` prevents landmark flicker from splitting
one exchange into multiple events.

The thresholds are learned from both c2 variants and c5 hard negatives. Do
not hard-code frame 81, frame 124, bbox expansion, paper visibility, or wrist
movement as the general rule. Object detection may confirm paper identity,
but it is not required for this pose/hand c2 proxy until object QA passes.

### Common separation from other classes

```text
c2 = peer-directed attention + directed hand exchange geometry
c3 = peer-directed attention + torso/face lean, without exchange geometry
c7 = peer-directed attention + hand gesture, without exchange geometry
c5 = no sustained peer-directed exchange; normal independent motion
c1/c4 = downward hand retraction/head-down desk behavior
```

The c2 rule is not satisfied by face turning alone, a bbox becoming large,
one wrist jump, or one point entering the margin without temporal and pair
support.



## c2 evidence: two-actor paper transfer candidate

The supplied c2 example is `VID20260804142017.mp4`. Its JSON is:

```text
data/raw_video/processed/holistic_outputs/VID20260804142017.json
```

The manifest marks both actors as sources:

```text
{"source":"s15","peer":"s16"}
{"source":"s16","peer":"s15"}
```

both actor tracks receive c2 evidence; not a single-target
video label. The JSON contains 376 source frames, 1920x1080 video geometry, and
approximately 29.92 FPS.

At the normal prefix, track 2 is the left image-space actor and track 1 is the
right image-space actor. Under the numeric actor rule this maps `s15 -> track 2`
and `s16 -> track 1`. Do not infer `track 1 = smaller s` from track numbering.

### JSON evidence from the supplied c2 example

Using source-frame indexing and a baseline of frames 0-80:

```text
frame 0-80: no confirmed transfer signal
frame 81:    track 1 head-turn onset candidate
frame 101:   track 1 head turn is strong
frame 124:   transfer/proximity peak candidate
```

For track 1, the baseline-relative pose changes are:

```text
frame 81:  delta nose_lateral = -0.0364, delta eye_lateral = -0.0289
frame 101: delta nose_lateral = -0.2503, delta eye_lateral = -0.2596
frame 124: delta nose_lateral = -0.4993, delta eye_lateral = -0.5231
```

At source frame 123 (`frame_id=124`), the current JSON gives bboxes
`track 1=[922,164,1600,699]` and `track 2=[282,297,916,724]`. Their union
spans approximately 68.6% of frame width and their bbox IoU is `0.0` because
there is a small horizontal gap. The previous `.018` IoU value is outdated.
The previous `.039` hand distance is also outdated for pair comparison: it was
computed from crop-local coordinates. Pair distances must use `frame_x/frame_y`
or a correct crop-to-frame projection. Using frame coordinates, the minimum
distance across all valid hand points at this frame is about 320 px (0.167 of
frame width); this broad minimum is a support diagnostic, not the c2 rule.
These measurements support a hand-transfer/proximity candidate, but they do
not prove that a paper was transferred; object identity remains a separate
detector concern.

At this peak, track 1 has `face_valid=false` while track 2 has
`face_valid=true`. Therefore face/head evidence must be accumulated before the
peak and must not be required to be valid on the exact transfer frame.

Track 2 hand lifting is supporting evidence. It must not be the primary c2
signal because the transfer can be represented by the pair's hand convergence
and the source/partner temporal order or sliding the paper across.

### Proposed c2 definition
The candidate definition is:

```text
both actors participate in an ordered exchange:
head/attention orientation toward the partner
-> hands converge across the pair boundary
-> one or both actors maintain transfer proximity
```

This is a visual hand-transfer proxy. It must not claim paper identity unless
an independently QA'd object detector confirms it.

### c2 pose and pair parameters

Use these pose points per actor:

```text
0       nose
1, 4    left/right eye support
11, 12  shoulders
13, 14  left/right elbows
15, 16  left/right wrists
23, 24  hips
```

Define:

```text
W_a,t = distance(pose[11], pose[12])
S_a,t = (pose[11] + pose[12]) / 2
H_a,t = (pose[23] + pose[24]) / 2
nose_lateral_a,t = (pose[0].x - S_a,t.x) / W_a,t
hand_set_a,t = valid pose elbows/wrists plus valid hand landmark points,
               represented in frame coordinates before pair comparison
```

The head feature is the signed peer-directed change in `nose_lateral`, with
eyes 1/4 as support. Hand features are:

```text
selected_exchange_point_a,t = highest-priority valid inward point
                              (finger -> wrist -> elbow -> shoulder)
hand_pair_distance_t       = frame-space distance between selected points
                             of both actors
hand_convergence_t         = baseline_hand_pair_distance - hand_pair_distance_t
hand_velocity_a,t          = hand displacement / W_a,t / dt
near_midpoint_pre_cross_t  = selected priority point enters the fixed 10% margin
cross_midline_t             = optional later crossing, not the primary flag
```

`selected_exchange_point` enforces the priority order. `hand_pair_distance`
measures proximity between the selected points. `hand_convergence` measures
movement toward the transfer zone. `hand_velocity` captures the rapid
reach/receive motion. Elbows 13/14 and wrists 15/16 provide arm-extension and reach
support; they do not replace the hand-finger points. `near_midpoint_pre_cross`
is the primary flag and means that a point from the highest available priority
tier entered the fixed pre-midpoint margin. `cross_midline` is optional later
evidence; neither flag proves contact or object transfer. The selected point
is the valid inward point closest to the fixed midpoint within the highest
available priority tier; anatomical left/right labels must not be hard-coded
across videos.

BBox support features are:

```text
bbox_union_fraction_t = union width of both bboxes / frame width
bbox_iou_t            = bbox intersection / bbox union
center_distance_t     = distance between bbox centers / frame width
pair_shared_zone_t    = overlap or near-contact of the two actor regions
```

`bbox_union_fraction` captures expansion toward the center. `bbox_iou` must
not be required to be high: two hands can approach while the person bboxes
remain mostly separate, as in both examples. `center_distance` and
`pair_shared_zone` are supporting geometry only because bbox expansion can
also be caused by tracking error.

Velocity parameters are:

```text
v_head_peer_a,t = change in signed peer-directed head displacement / dt
v_hand_a,t      = hand displacement / W_a,t / dt
```

`T_head_velocity` catches the rapid head-turn onset near frame 81.
`T_hand_velocity` catches reach/receive motion. Both thresholds are learned
from c5 false-positive motion and must be paired with displacement and
persistence thresholds; velocity alone is not c2. The fixed midpoint and
10% margin are computed once from the midpoint-baseline frame.

Temporal parameters are:

```text
L_head_to_hand = hand-convergence onset - head-turn onset
N_hand         = consecutive frames satisfying hand proximity/convergence
N_shared       = overlap duration of both actors in the shared zone
G_merge        = maximum gap when merging one transfer episode
```

`L_head_to_hand` measures exchange order. `N_hand` and `N_shared` reject a
single-frame crossing. `G_merge` prevents detector flicker from splitting one
transfer. These values must be learned/validated from c5 negatives and
confirmed c2 intervals, not guessed from frame 124 alone.

### Proposed c2 Stage 1 / Stage 2 path

Stage 1 emits `behavior_candidate` only when head orientation, hand
convergence, pair geometry, and temporal order jointly pass c5-derived
thresholds. It emits `normal` or `uncertain` otherwise.

Stage 2 classifies c2 after Stage 1 confirms the transfer candidate. The score
should combine head/eye direction, hand distance/convergence, hand velocity,
bbox geometry, and timing lag. Both source actors must be evaluated separately
and also as one pair.


## c2 variant 2 evidence: inner-hand midpoint and paper exchange

The second supplied c2 example is:

```text
1786174198374_53534665179931303_9030090460038115400_annotated.mp4
JSON: data/raw_video/processed/holistic_outputs/1786174198374_53534665179931303_9030090460038115400.json
```

The manifest marks `s19` and `s20` as sources, with both interaction pairs
present. The spatial mapping is `s19 -> track 1 -> left` and
`s20 -> track 2 -> right`; this is the numeric-actor ordering rule, not a
per-video identity guess. The JSON has 404 source frames (`source_frame_index`
0-403); `frame_id` is one-based. The manifest interval is only a coarse
relative bound and is not an exact event interval.

The user annotation is consistent with a c2 candidate, with these caveats:

```text
frames 0-48: normal prefix
frame 48:    action-onset candidate; verify with transition features
frame 59:    reported wrist discontinuity, but wrist0 is not discontinuous
frames 61-65: larger track-1 right-wrist change appears in JSON
frame 81:    key exchange geometry and face-turn candidate
frame 124:   later exchange/proximity keyframe
```

At source frame 80 (`frame_id=81`), the per-frame bbox-center midpoint would
be approximately `x=578.5 px`. The c2 rule does not use this recalculated
midpoint. It uses the fixed midpoint from the first valid paired frame. The
relevant *inner* finger points selected by image-space proximity are:

```text
left actor  (s19, track 1): hand point index 12, frame_x = 540.6 px
right actor (s20, track 2): hand point index 11, frame_x = 582.7 px
```

Relative to the per-frame diagnostic midpoint, neither point is strictly past
the opposite side: point 12 is 37.9 px left and point 11 is 4.2 px right. The
JSON therefore confirms inner-finger approach/convergence, not an unambiguous
crossing. Under the common rule, either point entering the fixed 10% pre-
midpoint margin is enough to raise the frame-level hand flag; crossing is not
required.

### c2 variant-2 hand feature contract

For this video, the following finger indices are the clearest observed inner-
hand cues:

```text
left actor  (smaller numeric s): inner finger point [12]
right actor (larger numeric s): inner finger point [11]
```

Here `[12]` and `[11]` mean raw MediaPipe hand-array finger indices. They are
not pose landmarks 11/12, which are shoulders. The hand array (`left_hand` or
`right_hand`) is selected dynamically as the valid hand whose finger point is
closest to the fixed midpoint on the actor's inward side. `frame_x/frame_y`
must be used, or crop-normalized `x/y` must first be projected through
`crop_bbox_xyxy`; never compare crop-local x values from two different tracks.

The words “12 phải + 11 trái” describe the two inward image-space hand cues
in this example; they must not be compiled as fixed anatomical field names.
Across videos, the anatomical hand can differ or MediaPipe handedness can
change under pose, occlusion, or crossing. The common implementation must
first map actors by numeric `s` order to image-space left/right, then select
the valid inward hand point closest to `pair_mid_x`; indices 11/12 remain the
preferred finger support points.

Define the fixed midpoint once, from the first paired-valid frame:

```text
actor_center_x_a,0 = (bbox_x1_a,0 + bbox_x2_a,0) / 2
pair_mid_x_0       = (actor_center_x_left,0 + actor_center_x_right,0) / 2
actor_gap_0        = actor_center_x_right,0 - actor_center_x_left,0
M_mid              = 0.10 * actor_gap_0

inner_left_x_t     = frame_x(inner finger point [12]) of the left actor
inner_right_x_t    = frame_x(inner finger point [11]) of the right actor

gap_left_mid_t     = pair_mid_x_0 - inner_left_x_t
gap_right_mid_t    = inner_right_x_t - pair_mid_x_0
inner_approach_t   = baseline gap - current directional gap
near_midpoint_pre_cross_t =
    (pair_mid_x_0 - M_mid <= inner_left_x_t <= pair_mid_x_0)
    OR (pair_mid_x_0 <= inner_right_x_t <= pair_mid_x_0 + M_mid)
inner_gap_t        = |inner_left_x_t - inner_right_x_t|
inner_velocity_t   = frame displacement of the two inner points / actor scale / dt
```

`pair_mid_x_0` is the camera-image midpoint between the two actor bbox centers
at the first paired-valid frame and is fixed afterward. `M_mid` is 10% of the
initial actor-center gap. `gap_left_mid` and `gap_right_mid` are directional
distances to this fixed midpoint. `inner_approach` is a decrease in either
directional gap relative to the normal baseline. `near_midpoint_pre_cross` is
true when either selected finger is within the 10% margin but remains on its
own side; it does not require crossing and can raise the frame-level flag
immediately. `inner_gap` measures hand convergence. `inner_velocity` captures
rapid reach/receive motion. Actor scale is the actor bbox diagonal or shoulder
width, selected consistently across videos. All x comparisons use camera
coordinates, where x increases to the right.

Do not use either hand wrist (`hand_landmarks[0]`) as the primary finger
feature. If finger points are missing, fall back in order to pose wrists
`16/15`, elbows `14/13`, and shoulders `12/11`. In this example the wrists
can be close while the fingers provide the useful left/right separation; wrist
proximity is only a fallback/support feature when fingers are valid.
Do not treat a single-frame wrist jump as action evidence. The JSON does not
show a track-1 wrist0 jump at source frame 59; the more visible change starts
around source frames 61-65 and must still be checked against hand-point
continuity and bbox/crop changes.

### c2 variant-2 candidate rule

For this variant, the candidate hierarchy is:

```text
looking_toward_peer
AND (near_midpoint_pre_cross OR sustained inner_approach)
AND inner_gap/convergence passes a c5-derived threshold
AND hand velocity/displacement persists for N_inner frames
AND pair geometry is valid
```

`N_inner` is the minimum consecutive-frame persistence for the inner-hand
signal. `T_inner_gap` is the maximum acceptable distance between the two
inner points after actor-scale normalization. `T_inner_velocity` is the
minimum normalized reach/receive speed. `M_mid_ratio=0.10` is the fixed
pre-midpoint margin ratio. These parameters must be learned from c2 examples
and c5 nearby-hand negatives; the midpoint itself must come only from the
first paired-valid frame and must not be recomputed per frame.

This variant is a stronger c2 hand-transfer proxy than a generic hand-motion
feature. It still does not prove paper identity or successful transfer. Both
source actors must receive actor-level c2 evaluation; pair-level aggregation
is additional, not a replacement for classifying both actors.




## Cue hierarchy revision: static face ROI and class separation

The strongest cues should be separated by behavior family rather than giving
all classes the same weight:

```text
c2     face/attention turn + hand reaches into shared partner zone
c3     face/attention turn + upper-body lean, without hand reach
c7     face/attention turn + hand gesture, without c2 transfer geometry
c1/c4-v1 hand retracts downward toward the desk; head-down is supporting evidence
c4-v2 hand/arm moves toward opposite sleeve; paired-limb proximity/overlap is primary
```

The c7 row is now supported by a first hard-case example below, but remains a
candidate rule until more c7 examples and c5 hand-motion negatives are tested.

### Static face ROI proposal

For each actor track, create a camera-space face ROI from a confirmed normal
prefix. Do not recenter this ROI on every frame, because that would erase the
signal being measured. Let:

```text
C_face = median face-anchor center over normal baseline frames
B_face = baseline face-anchor box over normal baseline frames
M_face = c5-derived expansion margin
ROI_face_static = expand(B_face, M_face)
```

The preferred pose anchors are nose `0` and eyes `1,4`; valid face-mesh anchors
may enlarge the box but may not replace missing pose points. `M_face` is learned
from c5 normal motion, not selected by visual inspection. If the camera moves,
estimate and remove camera motion first; otherwise a global static ROI will
confuse camera motion with actor behavior.

Define camera-view direction using the peer position:

```text
face_exit_peer_t = face anchor exits ROI_face_static
                   toward the peer's image-space direction
v_face_peer_t     = peer-directed face-anchor displacement / dt
looking_toward_peer_t = face_exit_peer_t
                        AND v_face_peer_t >= T_face_velocity
```

`looking_toward_peer` is a head-position/orientation proxy, not true gaze. It
requires `N_face_exit` consecutive frames or a learned hysteresis rule. A one
frame ROI exit is uncertain, not a class decision.

### Class decision candidates

After Stage 1 confirms `looking_toward_peer`, use class-specific supporting
features:

```text
c2_candidate = looking_toward_peer
                AND hand_reach_shared_zone
                AND pair_proximity/convergence

c3_candidate = looking_toward_peer
                AND torso_lean_peer
                AND no sustained hand_reach_shared_zone

c7_candidate = looking_toward_peer
                AND hand_gesture
                AND NOT c2_candidate

c1_c4_v1_candidate = hand_retract_downward
                     AND head_down_support

c4_v2_candidate = opposite_sleeve_motion
                  AND paired_limb_proximity_or_overlap
```

`hand_reach_shared_zone` means a valid wrist/hand point moves toward or across
the midpoint between actors with sufficient velocity and persistence.
`pair_proximity/convergence` uses hand distance, bbox union, and center
distance; bbox overlap alone is not contact evidence. `torso_lean_peer` uses
the signed shoulder/hip torso-lateral displacement already defined for c3.
`hand_gesture` means a sustained hand motion/shape event that does not enter
the c2 shared transfer zone. The first c7 hard case below supports a gesture
signal based on hand/finger motion or a raised/formed hand remaining on the
actor's own side; it does not support using hand motion alone without the
no-exchange condition.
`hand_retract_downward` is the primary c1/c4-v1 cue: normalized wrist/elbow
displacement and downward velocity from the normal desk baseline. For c4-v2,
`opposite_sleeve_motion` is represented by actor-scale-normalized proximity or
overlap of `(16,15)`, `(14,13)`, or `(12,11)`; `head_down` is supporting evidence
when present, not a replacement for the paired-limb cue. Nose `0` belongs only
to the head-down calculation.

The classifier must preserve `uncertain` when the face ROI, hand validity, or
pair geometry is missing. It must not convert a failed ROI test into c5, and it
must not call a hand/face proxy paper transfer, speech, or true gaze.




## c7 hard case 1: hand signal without c2 exchange geometry

The supplied c7 example is:

```text
1786174214580_53534665179931303_9030090460038115400_annotated.mp4
JSON: data/raw_video/processed/holistic_outputs/1786174214580_53534665179931303_9030090460038115400.json
```

The manifest contains two source actors:

```text
source s21
source s22
actor_ids         = ["s21", "s22"]
action_actor_ids  = ["s21", "s22"]
class_code        = c7
interaction_pairs = [
  {"source": "s21", "peer": "s22"},
  {"source": "s22", "peer": "s21"}
]
```

The JSON contains 509 source frames (`source_frame_index` 0-508), with
`frame_id` one-based. At source frame 0, track 1 is the left actor and track 2
is the right actor, so the numeric mapping is `s21 -> track 1` and
`s22 -> track 2`.

### Timeline and JSON evidence

The supplied annotation is:

```text
frames 0-44: normal prefix
frame 56:    track 1 head tilt/tap signal candidate
frame 80:    track 1 hand raise candidate
frame 90:    track 2 head-turn candidate
frame 99:    track 1 hand gesture candidate
frame 106:   track 1 stronger head-turn candidate
frame 129:   track 2 hand raise candidate
frame 144:   track 2 hand sign candidate
frame 298:   track 2 raised sign candidate
frame 427:   track 1 raised-hand candidate
```

The JSON supports peer-directed head movement as a proxy, not true gaze. For
track 1, `nose_lateral` changes from `0.017` at source frame 0 to `0.175` at
frame 99 and `0.206` at frame 106. For track 2 it changes from `-0.036` at
frame 0 to `-0.164` at frame 90. These values support a head-orientation cue,
but not a speech or gaze claim.

The c2 fixed midpoint audit is:

```text
track 1 center at frame 0 = 464.0 px
track 2 center at frame 0 = 781.5 px
pair_mid_x_0              = 622.75 px
actor-center gap D_actor_0 = 317.5 px
M_mid = 10% * D_actor_0    = 31.75 px
```

The minimum finger-point distance to `pair_mid_x_0` remains outside the c2
margin at the annotated gesture frames:

```text
source frame        56    80    90    99   106   129   144   298   427
track 1 min |x-mid| 50.5  47.1  49.7  68.8  69.8  81.1  78.5  79.3  37.1
track 2 min |x-mid| 55.3  56.9  53.5  51.8  51.0  53.0  56.5 138.3  49.4
margin              31.75 px for both actors
```

Therefore the hand signals do not satisfy c2's
`near_midpoint_pre_cross` flag. Frame 427 track 1 is the closest case at
`37.1 px`, only `5.35 px` outside the margin; it is a c2 hard negative and
must not be silently relabeled as c2.

The hand geometry is consistent with own-side gestures rather than exchange
geometry. Examples include track 2's left-hand region moving to approximately
`x=842.9-904.8, y=615.7-651.2` at frame 144 and
`x=761.0-804.4, y=562.6-611.4` at frame 298, while remaining far from the
fixed midpoint. Track 1's left-hand region at frame 427 is approximately
`x=524.5-585.6, y=558.1-598.7`, also outside the c2 margin.

There is one annotation/JSON discrepancy: at source frame 0, track 2's
left-hand region is already high (`y=456.0-510.8`), then appears lower by
frame 44 (`y=597.6-625.3`), although frames 0-44 are annotated normal. A raw
absolute hand-y threshold would therefore create a false c7 onset. Use a
robust normal-prefix baseline and hand continuity/shape evidence; do not treat
frame 0 as a guaranteed normal hand-height sample.

### c7 candidate rule from this hard case

At pair level:

```text
pair_has_two_valid_actors
AND face_or_head_turn_toward_peer
AND hand_gesture_signal
AND NOT near_midpoint_pre_cross
AND NOT sustained hand convergence/shared-zone proximity
```

`hand_gesture_signal` may be a finger configuration/motion, a raised hand, or
a tap/hand signal that remains on the actor's own side. It must be measured
relative to a robust normal-prefix baseline, not frame 0 alone. The no-c2
condition is mandatory: a hand gesture that enters the fixed 10% midpoint
margin belongs to the c2 decision path until further evidence separates it.

### c7 parameter definitions

```text
T_raise            minimum upward hand/finger displacement from normal baseline
T_gesture_velocity minimum hand/finger motion speed for a dynamic signal
N_gesture          persistence frames for gesture evidence
T_shape_change     minimum stable finger-configuration change, if detectable
T_head_peer        minimum peer-directed head displacement for looking support
T_face_velocity    minimum peer-directed face-anchor velocity
N_face_exit        persistence frames for static face-ROI exit
T_face_valid       maximum local face-invalid ratio for mandatory face support
T_hand_valid       minimum hand validity for occluded-face override
M_mid              c2 fixed midpoint margin; 10% of initial actor-center gap
T_own_side         minimum distance from fixed midpoint to remain an own-side cue
G_gesture          maximum gap when merging gesture frames into one episode
```

`T_raise` uses image y, where decreasing y means upward motion, normalized by
shoulder width or actor bbox scale. `T_gesture_velocity` captures tapping or a
rapid sign transition. `N_gesture` rejects a one-frame detector artifact.
`T_shape_change` is optional until hand landmark visibility and handedness QA
are complete. `T_head_peer` measures normalized head displacement toward the
peer. `T_face_velocity` and `N_face_exit` apply to the static face-ROI signal.
`T_face_valid` decides whether face support is sufficiently observed locally;
it is not a behavior negative. `T_hand_valid` protects the occluded-face
fallback from missing hand data. `M_mid` is not learned per frame; it is fixed
from the first paired-valid frame under the c2 contract. `T_own_side` prevents
a gesture near the pair boundary from being mislabeled c7. `G_gesture` merges
fragmented gesture frames. All thresholds require c5 normal hand-motion
negatives and more c7 examples.

### c7 calculation formulas

Let `B_a` be the annotated normal-prefix frames for actor `a`, not necessarily
only frame 0. Use a robust median and MAD over `B_a`:

```text
baseline(q,a) = median_{t in B_a}(q_a,t)
MAD(q,a)      = median_{t in B_a}(|q_a,t - baseline(q,a)|)
S_a,t         = shoulder width, or bbox diagonal if shoulders are invalid
```

The hand/arm point group is selected in this strict order:

```text
G_finger = valid hand-array finger landmarks, excluding hand wrist index 0
G_wrist  = pose[16], pose[15]
G_elbow  = pose[14], pose[13]
G_shoulder = pose[12], pose[11]
G*_a,t = first group with sufficient valid points in the current window
```

`G_shoulder` is only a last fallback. It is never a primary c7 gesture cue.
For every point `q` in the selected group, calculate upward displacement and
speed in camera coordinates (`y` increases downward):

```text
raise_q,a,t = (baseline(q_y,a) - q_y,a,t) / S_a,t
speed_q,a,t = sqrt((q_x,a,t-q_x,a,t-1)^2 +
                   (q_y,a,t-q_y,a,t-1)^2) / (S_a,t * dt)

raise_a,t = max_{q in G*_a,t}(raise_q,a,t)
speed_a,t = max_{q in G*_a,t}(speed_q,a,t)
```

If the selected group is finger landmarks, estimate hand-shape change using
finger geometry. Hand wrist index 0 is only a normalization reference here,
not a c2 crossing cue:

```text
P_a,t = distance(hand[0], hand[9])
z_a,t = [ distance(hand[0],hand[4]),
          distance(hand[0],hand[8]),
          distance(hand[0],hand[12]),
          distance(hand[0],hand[16]),
          distance(hand[0],hand[20]) ] / P_a,t

shape_change_a,t = ||z_a,t - median_{u in B_a}(z_a,u)||_2 / sqrt(5)
```

`raise` detects a hand moving upward from the normal desk posture. `speed`
detects a tap or rapid sign transition. `shape_change` detects a formed finger
configuration when the hand remains approximately in place. A shape score is
unknown, not zero, when the required finger points are occluded.

Use the fixed c2 midpoint to enforce own-side geometry:

```text
side_a = -1 if actor a is left of pair_mid_x_0 else +1
x_a,t  = frame_x of the selected inward gesture point
own_side_dist_a,t = side_a * (x_a,t - pair_mid_x_0)

near_midpoint_pre_cross_a,t =
    0 <= own_side_dist_a,t <= M_mid

own_side_c7_a,t = own_side_dist_a,t > max(M_mid, T_own_side)
```

`own_side_dist` is positive when the point remains on its actor's own side.
`near_midpoint_pre_cross` is the c2 exclusion. `own_side_c7` ensures that a
gesture near the pair boundary is not called c7 merely because it has not yet
crossed.

The frame-level hand gesture and c7 flags are:

```text
hand_gesture_frame_a,t = own_side_c7_a,t
                          AND (
                              raise_a,t >= T_raise
                              OR speed_a,t >= T_gesture_velocity
                              OR shape_change_a,t >= T_shape_change
                          )

head_peer_a,t = peer_direction_a *
                (nose_x,a,t - baseline(nose_x,a)) / S_a,t

pose_head_support_a,t = head_peer_a,t >= T_head_peer
face_head_support_a,t = face_exit_peer_a,t
                       AND face_velocity_peer_a,t >= T_face_velocity
                       AND face_exit_persistence_a,t >= N_face_exit
head_support_a,t = pose_head_support_a,t OR face_head_support_a,t

c7_frame_a,t = valid_actor_a,t
                AND hand_gesture_frame_a,t
                AND NOT near_midpoint_pre_cross_a,t
                AND NOT shared_zone_convergence_a,t
                AND (
                    head_support_a,t
                    OR occluded_face_with_strong_gesture_a,t
                )
```

`peer_direction_a` is `+1` for an actor whose peer is to the camera-right and
`-1` when the peer is to the camera-left. `head_support` is optional only when
face occlusion is measured. Face is primary when valid; pose head displacement
is the fallback support and must not be interpreted as true gaze. Define the
occlusion override explicitly:

```text
local_face_valid_a,t = valid face frames / frames in the gesture window
occluded_face_with_strong_gesture_a,t =
    local_face_valid_a,t < T_face_valid
    AND hand_gesture_frame_a,t persists for N_occluded_gesture frames
    AND hand_validity_a,t >= T_hand_valid
```

Finally, merge frame flags into an event:

```text
c7_episode_a =
    consecutive_or_gap_merged(c7_frame_a,t,
                              min_frames=N_gesture,
                              max_gap=G_gesture)
```

`N_gesture` and `G_gesture` prevent one noisy hand frame from becoming c7.
The frame-level flag can be emitted immediately for inspection, but the
episode-level c7 decision must pass persistence or return `uncertain`.

The two source actors still receive actor-level c7 labels from
`action_actor_ids`; pair-level c7 detection must not classify only the actor
whose gesture is visually strongest.


## c7 hard case 2: gesture under heavy track-2 occlusion

The second supplied c7 example is:

```text
WIN_20260802_12_01_58_Pro_annotated.mp4
JSON: data/raw_video/processed/holistic_outputs/WIN_20260802_12_01_58_Pro.json
```

The manifest contains two source actors:

```text
source s11
source s12
actor_ids         = ["s11", "s12"]
action_actor_ids  = ["s11", "s12"]
class_code        = c7
interaction_pairs = [{"source":"s11","peer":"s12"},
                      {"source":"s12","peer":"s11"}]
```

The JSON contains 474 source frames (`source_frame_index` 0-473), with
`frame_id` one-based. Track 2 is the left image-space actor and track 1 is the
right actor, so the numeric mapping is `s11 -> track 2` and `s12 -> track 1`.

### Timeline and occlusion audit

The supplied annotation is:

```text
frames 0-40: normal prefix
frame 41:    track 2 hand-raise candidate
frame 52:    track 1 head-turn onset candidate
frame 87:    track 1 hand normal
frame 92:    track 1 begins forming a hand sign
frames 52-129: track 1 turns fully while hand remains normal for much of it
frame 137:   track 1 hand-sign candidate
frame 290:   track 2 hand-sign candidate
frame 352:   track 1 hand-sign candidate
```

The occlusion is a real quality constraint, not a negative behavior signal:

```text
track 1: face_valid 282/474; left hand 470/474; right hand 467/474
track 2: face_valid 21/474;  left hand 465/474; right hand 134/474
```

Track 2's face is therefore unavailable in most frames and its right hand is
missing in most frames. A failed face ROI or missing right-hand array must not
be converted to c5. For track 2, use the valid left-hand/finger stream and
pose fallback, while preserving `uncertain` when the remaining evidence is
insufficient.

### Fixed c2 midpoint rejection

Using the first frame's actor centers:

```text
left actor  track 2 center x = 567.5 px
right actor track 1 center x = 1370.0 px
pair_mid_x_0                 = 968.75 px
D_actor_0                    = 802.5 px
M_mid = 10% * D_actor_0      = 80.25 px
```

The closest valid finger point to the fixed midpoint remains outside the c2
pre-midpoint margin at all supplied gesture markers:

```text
source frame        41    52    87    92   129   137   290   352
track 1 min |x-mid| 354.6 353.0 350.9 351.5 356.9 330.8 378.9 363.5
track 2 min |x-mid| 333.0 331.3 334.1 337.5 343.1 342.8 316.7 309.9
margin               80.25 px for both actors
```

This is strong evidence against c2 shared-zone exchange. At frame 137,
track 1's right-hand region is approximately
`x=1299.6-1354.8, y=572.5-616.9`; at frame 352 it is
`x=1332.3-1384.3, y=539.5-582.9`. Track 2's left-hand region is approximately
`x=625.8-663.4, y=585.4-620.6` at frame 290. These gestures remain on each
actor's own side and are hundreds of pixels from the fixed midpoint.

### c7 rule refinement for occluded actors

For a valid pair:

```text
pair_has_two_valid_actors
AND hand_gesture_signal
AND NOT near_midpoint_pre_cross
AND NOT sustained hand convergence/shared-zone proximity
AND (
    peer-directed head/face support is valid
    OR face is occluded but gesture evidence is valid and persistent
)
```

The face/head clause is optional only under measured occlusion. It is not an
excuse to ignore face quality: the output must carry an occlusion/uncertainty
flag and should require stronger hand persistence or shape evidence. In this
case, track 1's head turn during frames 52-129 must not by itself become c7,
because the hand is annotated normal for much of that interval. Track 1's
later hand sign and track 2's own-side hand sign are the subtype evidence.

The two source actors still receive actor-level c7 labels from
`action_actor_ids`. A missing face or hand on one source actor changes
confidence; it does not turn that source into a peer-only c5 actor.


## c3 evidence: peer-directed look/peek candidate

The supplied c3 example is `VID20260804141339.mp4`. Its JSON is:

```text
data/raw_video/processed/holistic_outputs/VID20260804141339.json
```

The user identifies track 1 as the action source and track 2 as the peer. The
JSON contains 394 source frames (`source_frame_index` 0-393), with `frame_id`
being one-based. The source track has valid face landmarks; the peer track has
no valid face landmarks in this output. The c3 decision therefore uses source
track pose/face and peer geometry, while peer face is only a validity signal and
must not be treated as a missing negative.

action_actor_ids = ["s16"]
interaction_pairs = [{"source":"16","peer":"s15"}]


### JSON evidence from the supplied c3 example

Using source-frame indexing and the pose shoulder-normalized signals defined
below, track 1 has the following baseline and changes. The baseline is the
median of source frames 0-10:

```text
baseline nose_lateral = -0.01688
baseline nose_d       = -0.17321
baseline mouth_d      = -0.12081
baseline eye_d        = -0.27837

source frame 12:  delta nose_lateral = -0.0858, delta nose_d = -0.0307
source frame 27:  delta nose_lateral = -0.3597, delta nose_d = -0.2423
source frame 49:  delta nose_lateral = -0.3439, delta nose_d = -0.3034
source frame 326: delta nose_lateral = -0.2164, delta nose_d = -0.1382
```

The JSON supports a progressive turn beginning near frame 12, a strong turn
near frame 27-49, and a later moderate but sustained turn near frame 326. C3
must not require an extreme turn: frame 326 is important evidence that a
moderate peer-directed turn can still be a candidate when persistence and
direction are correct. frame 326 is a separate c3 event 

### Proposed c3 definition

The current candidate definition is:

```text
source actor turns head and/or upper body toward the peer
for a sustained interval, with the turn direction agreeing with peer position
```

This is a peer-directed visual-orientation proxy, not a claim of true gaze or
intent. A single head rotation is insufficient.

For source actor `a` at frame `t`, define:

Pose points used by c3:

```text
0       nose             primary head-orientation point
1, 4    left/right eye   support for nose direction
11, 12  left/right shoulder  shoulder midpoint and shoulder scale
23, 24  left/right hip       hip midpoint and torso scale
```

The required c3 pose set is `{0, 1, 4, 11, 12, 23, 24}`. Nose `0` is primary;
the eye midpoint from `1,4` supports the nose direction. If a required point is
missing or invalid, the corresponding head/torso measurement is unknown; it
must not be replaced with zero. Pose points `9,10` may be logged as optional
mouth corroboration, but they are not required for the c3 decision. Hand
landmarks and face-mesh landmarks are not c3 primary signals.

For the front two-actor layout, direction is tied to numeric actor order, not
to the raw track ID:

```text
peek direction is always defined from the camera/image viewpoint
image x decreases -> camera-left
image x increases -> camera-right

s_small (left in image) -> peer on camera-right: peek direction = camera-right
s_large (right in image) -> peer on camera-left: peek direction = camera-left
```

In the project convention, a rightward image-space lean/shift of pose point
`11` for the left/smaller actor is evidence of that actor peeking toward the
camera-right peer. The inverse interpretation applies to the right/larger
actor and pose point `12`: a leftward image-space lean/shift is evidence of
peeking toward the camera-left peer. These are camera-view directions, not the
actor's anatomical left/right labels. The actor-to-track mapping must be
resolved first; landmark index alone must never be used as actor identity.

```text
W_a,t = distance(pose[11], pose[12])
S_a,t = (pose[11] + pose[12]) / 2
H_a,t = (pose[23] + pose[24]) / 2
T_a,t = distance(S_a,t, H_a,t)
nose_lateral_a,t = (pose[0].x - S_a,t.x) / W_a,t
torso_lateral_a,t = (S_a,t.x - H_a,t.x) / T_a,t
```

`nose_lateral` measures head orientation relative to the shoulder midpoint.
`torso_lateral` measures upper-body lateral displacement relative to the hip
midpoint. `W` is shoulder scale. `T` is torso scale. All coordinates must come
from the same normalized pose coordinate system.

Let `peer_mid_x` be the peer track's bbox or shoulder midpoint x coordinate and
let:

```text
peer_direction = sign(peer_mid_x - S_a,t.x)
```

Baseline-relative peer-directed signals are:

```text
delta_head_peer_t  = peer_direction *
                      (nose_lateral_a,t - median_baseline(nose_lateral_a))

delta_torso_peer_t = peer_direction *
                      (torso_lateral_a,t - median_baseline(torso_lateral_a))
```

`delta_head_peer` is the primary c3 signal. `delta_torso_peer` is supporting
evidence for the upper-body turn. The sign makes the feature actor-position
independent: positive means movement toward the peer.

The frame-level candidate is:

```text
pose_valid
AND peer_track_valid
AND delta_head_peer_t >= T_head
AND delta_torso_peer_t >= T_torso  [supporting condition]
```

The supporting torso condition may be relaxed only when the source face/head
signal is strong and persistent. `T_head` and `T_torso` are learned from c5
normal frames and c5 false-positive turns; they are not fixed values.

Because the c3 turn can begin rapidly around frame 12, velocity is required as
an onset signal. It is computed from the signed peer-directed displacement:

```text
v_head_peer_a,t  = (delta_head_peer_a,t - delta_head_peer_a,t-1) / dt
v_torso_peer_a,t = (delta_torso_peer_a,t - delta_torso_peer_a,t-1) / dt
```

`v_head_peer` is the normalized rate at which the head turns toward the peer.
`v_torso_peer` is the corresponding upper-body rate. These are orientation
proxies, not true angular velocities. The onset candidate is:

```text
turn_onset_candidate_t =
    peer_track_valid
    AND v_head_peer_t >= T_head_velocity
    AND delta_head_peer_t >= T_head_onset_displacement
```

`T_head_velocity` catches the rapid turn onset. `T_head_onset_displacement`
prevents a high velocity caused by a tiny noisy coordinate change. Both
thresholds are learned from c5 false-positive turns. Velocity starts an
episode; it must not be used alone as the final c3 decision. The final decision
still requires `delta_head_peer`, optional torso support, and persistence
`N_head`/`N_overlap`.

The temporal parameters are:

```text
N_head       = consecutive frames satisfying the head candidate
N_torso      = consecutive frames satisfying the torso support
N_overlap    = overlap of head and torso candidate runs
G_merge      = maximum gap allowed when merging adjacent candidate runs
P_peak       = maximum delta_head_peer within one episode
```

`N_head` and `N_torso` control persistence. `N_overlap` prevents unrelated head
and torso movements from being joined. `G_merge` prevents detector flicker
from splitting one turn into many events. `P_peak` is descriptive, not a
required extreme threshold; this preserves the frame-326 moderate-turn case.
All persistence and gap values must be learned/validated against c5 false
runs and confirmed c3 intervals.

## Proposed c3 Stage 1 / Stage 2 path

Stage 1 detects a sustained peer-directed orientation transition:

```text
normal -> uncertain -> behavior_candidate
```

It must use the signed peer direction, head/torso persistence, validity masks,
and episode timing. It must not emit c3 directly and must not convert every
turn into c3.

Stage 2 may classify c3 only after Stage 1 confirms the candidate. The c3
subtype score should combine `delta_head_peer`, `delta_torso_peer`, `N_head`,
`N_overlap`, episode duration, and `P_peak`. A moderate sustained turn can
qualify; an isolated extreme frame cannot.

C3 should be separated from:

- c5: normal head/torso movement without a sustained peer-directed episode;
- c1/c4: head-down plus arm-departure pattern;

Before c3 enters training, corrected actor IDs in the manifest, explicit
frame-level intervals for the frame-12 episode and the frame-326 episode, c5
hard negatives for ordinary head turns, and actor-level evaluation for source
track 1 versus normal peer track 2 are required. The coarse manifest action
bounds do not replace these frame-level annotations.

## Threshold and frame persistence learned from c5

For each c5 training track, compute the same baseline-relative signals and build a false-positive mask. Do not choose `0.10`, `0.15`, or a fixed frame count without measuring c5:

```text
T_nose    = quantile_99.5%(delta_nose on c5 normal frames)
T_support = quantile_99.0%(median(delta_mouth, delta_eye) on c5 normal frames)

candidate_t =
    (delta_nose_t >= T_nose)
    AND (median(delta_mouth_t, delta_eye_t) >= T_support)
```

For every c5 video, compute the longest consecutive run of `candidate_t`. Let `R_v` be that run for video `v`:

```text
N_enter = P99({R_v for c5 videos}) + 1
```

The normal-to-behavior transition is confirmed only when `candidate_t` is true for `N_enter` consecutive frames. If the deployment FPS is known, also enforce a minimum duration:

```text
N_enter = max(P99({R_v}) + 1, ceil(0.4 * FPS))
```

For c1/c4, combine it with arm departure:

```text
head_run >= N_enter
AND arm_departure_run >= N_arm
AND overlap(head_run, arm_departure_run) >= N_overlap
```

The values `N_arm` and `N_overlap` must be learned with the same c5 false-run procedure and checked on confirmed c1/c4 transition intervals. Do not flag from one 30-frame window; the current temporal windows have length 30 and stride 20, so persistence should be evaluated at frame level before aggregation.

# Legacy timing note

For the example, baseline is frames `0–33`. A deployment implementation must use a safe initial baseline when a confirmed normal prefix is unavailable.

The fixed frame values previously listed here are deprecated. Use the c5-derived `N_enter`, `N_arm`, and `N_overlap` formulas above; do not copy fixed counts into the implementation.

```text
deprecated fixed counts; not implementation values
deprecated historical clear count; not implementation value
```

The old fixed-count formula below is retained only as historical context:

```text
deprecated historical formula; use the c5-derived formulas in the section above
```

Use frame-level persistence/debounce; do not flag from one 30-frame window. Current temporal windows use length 30 and stride 20.

## Implementation behavior: current questions, blockers, and phase status (2026-08-10)

This is the current implementation snapshot. The semantic rules and feature
definitions above remain the ground truth; this section records what is
confirmed, what is blocked, and what must be answered before the next change.

### Resolved implementation contract: `v_c4_s7`

The `v_c4_s7` audit is resolved from the canonical JSON and the demo rendered
by `pose_gaze/pose_gaze/holistic/test_media`. Use this contract for future
implementation work:

```text
source actor/track = s7 / track 1
peer actor/track = s8 / track 2
track mapping = stable; no occlusion or switch
normal baseline = source_frame_index 0-36
c4 variant = c4-v2
episode onset = source_frame_index 64
head_down = source_frame_index 68-219
paired sleeve cue = source_frame_index 64 through final frame
episode end = not observed; no return to normal before video end
```

For c4-v2, implementation must measure opposite-sleeve movement using
actor-scale-normalized proximity or overlap of paired pose landmarks:
`(16,15)`, `(14,13)`, and `(12,11)`. Nose `0` is head-down support only.
Do not replace paired relations with single-landmark `16`, `14`, or `12`
features. This contract is semantic ground truth; thresholds, persistence,
and event metrics remain unvalidated.

### Open questions requiring video/JSON confirmation

1. For `v_c4_s8`, which canonical actor/track is the c4 source actor? The
   answer must come from the canonical JSON overlay, not from the regenerated
   demo currently on disk.
2. Are canonical `source_frame_index` `0–29` frames a safe normal baseline for
   the c4 actor? If not, which exact prefix is safe? The first 30 frames are
   currently provisional, not accepted as universally valid.
3. For the c4 episode, what are the exact onset, peak, and end frame ranges?
   Which side moves first: wrist `15/16`, elbow `13/14`, or both? Does the
   wrist move downward in image coordinates, and does the elbow support it?
4. Does the head-down interval overlap the arm-departure interval? Record the
   overlap in canonical frame indices; the coarse manifest bounds are not
   sufficient.
5. Does the actor-to-track mapping remain stable through the episode, or is
   there an occlusion/track switch?
6. For the c5 `s24` run of 58 frames, confirm that looking at the camera during
   the first approximately 1.5 seconds is a valid c5 hard negative. It must
   not be removed merely because it destabilizes the threshold.

### Current blockers and known problems

1. **Canonical/demo measurement boundary.** Detector/MediaPipe reruns can
   differ from canonical JSON (the known `v_c4_s8` difference is up to
   approximately 88.9 px for track 1 and 136.9 px for track 2). This mismatch
   is acceptable for visual demo inspection when not materially changing the
   observed action, but exact frame/landmark measurements must use canonical
   JSON. For exact audit, render JSON through
   `pose_gaze/pose_gaze/holistic/test_media` with `--landmarks-input`; do not
   treat a fresh inference JSON as canonical.
2. **The c5 s24 head run of 58 is a real hard negative, not missingness.**
   It produces 97 train head-positive frames in 5 episodes; the longest run is
   58 and the raw rate is 8.86 frames/minute. This makes the learned
   `N_enter=46` dominated by one actor behavior and not acceptable as a
   deployment threshold.
3. **Positive coverage pose-only is incomplete.** Current raw frame counts are:

   | Class | Head-down frames | Arm-departure frames | Head + arm overlap |
   |---|---:|---:|---:|
   | c1 | `188 / 19,218` (0.98%) | 0 | 0 |
   | c2 | 0 | 71 | 0 |
   | c3 | 0 | 0 | 0 |
   | c4 | `87 / 22,728` (0.38%) | 0 | 0 |
   | c7 | 0 | 2 | 0 |

   These are raw pose-only threshold hits, not video/event recall. c1 has no
   arm-departure coverage; c4 has no arm-departure or shared head+arm episode
   under the old feature branch. For c4-v2, this zero does not disprove the
   behavior: c4-v2 requires opposite-sleeve movement plus paired proximity or
   overlap of `(16,15)`, `(14,13)`, or `(12,11)`, which the old
   `arm_departure` count does not measure.
4. **Arm thresholds are not validated.** Current fitted values are:

   ```text
   T_arm_disp = 1.2086
   T_hand_down = 0.5936
   T_nose = 0.5422
   T_support = 0.4418
   ```

   `N_arm=1` and `N_overlap=1` are degenerate because c5 arm/overlap runs are
   all zero. These values must not become deployment parameters. c4-v2 needs a
   separate paired-distance threshold audit; reusing `T_arm_disp` would be an
   unsupported feature substitution.
5. **Baseline validity is unresolved.** A fixed first-30-frame baseline can
   absorb an action or an actor looking at the camera. Each `(video, actor)`
   needs a baseline state: `valid`, `provisional`, or `invalid`, with a reason.
6. **The old generic Stage 1 detector was two-sided.** It treated both tails
   of head-depth deviation as abnormal, which explains false alerts when the
   head starts unusually high/deep and then returns toward zero. c1/c4 must
   use the signed directional head-down rule above, not generic absolute
   abnormality. c4-v2 additionally requires the paired-limb sleeve cue.
7. **No valid integrated behavior metric exists yet.** Stage 1 currently has
   partial diagnostics, but Stage 2/XGBoost has not been validated on the
   corrected actor-level candidates. No F1 improvement may be claimed.

### Implementation phase status

| Phase | Scope | Status | Evidence / reason |
|---|---|---|---|
| 0 | Contract, manifest, actor/source/peer semantics | Completed | Ground-truth rules recorded above; background excluded. |
| 1 | Actor ID to track mapping | `v_c4_s7` confirmed; `v_c4_s8` pending | Numeric `s` ordering maps left/right, then to actual tracks; `v_c4_s7` is `s7 -> track 1`, `s8 -> track 2`, stable through the episode. |
| 2 | Canonical frame features | Completed with QA correction | Normalized pose plus frame coordinates rebuilt; output header has 341 unique fields. |
| 3 | Temporal geometry primitives | Rebuilt / partial validation | c1/c4-v1 branch exists; c4-v2 paired-sleeve geometry is now specified from `v_c4_s7`, but thresholds and persistence remain provisional. |
| 4 | Stage 1 c5 normal envelope | **Blocked** | c5 hard negative creates run 58; `N_enter=46` is rejected for deployment. |
| 5 | c1/c4 candidate detector | **Blocked** | c4-v2 semantic cue is confirmed, but paired-limb thresholds, persistence, and actor-level recall are not implemented or validated. |
| 6 | c2 detector | Specified, not integrated | Hand/finger priority and fixed midpoint rule are recorded above. |
| 7 | c3 detector | Specified, not integrated | Signed peer direction, velocity onset, torso support, and persistence are recorded above. |
| 8 | c7 detector | Specified, not integrated | Gesture/hand movement must be separated from c2/c3 using actor-level evidence. |
| 9 | c6 | Excluded / pending | Semantics are not sufficiently reliable for the current manifest. |
| 10 | Stage 2 XGBoost subtype classification | Not started | No trustworthy Stage 1 candidates yet. |
| 11 | F1/confusion-matrix evaluation | Blocked | Must wait for actor-level candidate coverage and canonical provenance. |
| 12 | Demo/deployment handoff | Blocked | `v_c4_s7_demo.mp4` is available for visual review; deployment metrics and canonical threshold validation remain blocked. |

### Required next sequence

1. Keep `v_c4_s7` contract above as c4-v2 semantic ground truth; do not
   generalize it to c4-v1 without another canonical audit.
2. For `v_c4_s8`, use the JSON-only renderer and resolve its actor/track,
   baseline, episode, and mapping questions before combining it with `v_c4_s7`.
3. Audit raw c5 `s24` head values and mark the episode as a retained hard
   negative; do not lower thresholds to fit it away.
4. Audit c4-v2 paired distances at confirmed keyframes, plus c4-v1
   elbow/wrist displacement, image-y direction, validity, and head/arm overlap.
5. Only after those audits, revise baseline validity, paired-distance
   thresholds, directional thresholds, and persistence. Then rerun Stage 1
   metrics before starting Stage 2.

