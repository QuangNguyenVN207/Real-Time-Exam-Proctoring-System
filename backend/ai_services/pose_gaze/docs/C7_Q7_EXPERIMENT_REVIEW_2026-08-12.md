# c7 Q7 actor-score experiment review (2026-08-12)

## Scope

Primary unit is `(video, actor_id)`. This review contains no video-level
metric. The preserved comparison checkpoint is
`behavior_actor_only_c2_c3_c7_feature_families_v3` (c2=.923, c3=.667,
c7=.323, c5=.625, macro=.634).

## Experiment result

`behavior_actor_only_c2_c3_c7_q7_pair_actor_score_v1` used:

`Q7 = V_hand AND OwnSide AND NOT NearMidpoint AND NOT SharedZone AND (Raise OR ShapeChange)`

with `Raise = baseline-relative upward displacement AND directional upward
speed`, `S7 = max_Q7(p7 - p5)`, and explicit-pair-only c2/c7 propagation.
Train-only c5 P95 geometric gates were displacement=.4656, upward speed=.0361,
and shape=.8326. The train-only S7 threshold was .7513.

Actor F1: c2=.800, c3=.571, c7=.000, c5=.531, macro=.476. The result is a
regression and must not replace the preserved checkpoint.

## Evidence-backed failure mechanism

Among the 20 failed c7 actors, 19 had at least one Q7-qualified frame. Their
median qualified-frame count was 11 and median qualified fraction .030. Thus
Q7 did not generally eliminate their evidence. Their median S7 was -.2387;
the highest c7 S7 was .6844, below the train threshold .7513. Two c5 false
positives exceeded threshold at .8646 and .8240. The p7-p5 ordering is wrong
on held-out actors, so lowering the threshold from the test result would be
locked-test tuning and is prohibited.

## Open semantic defects to resolve before another experiment

1. The implementation currently chooses the point closest to the midpoint for
   `OwnSide`, averages all valid wrists for `Raise`, and computes shape across
   both hands. A single Q7 frame can combine own-side evidence from one hand
   with raise/shape evidence from another. Q7 needs one selected, persistent
   gesture-hand identity before its conjunction has the intended meaning.
2. `ShapeChange` is previous-frame shape difference, not change from the
   normal-prefix baseline. Occlusion/reacquisition can therefore qualify a
   frame without a gesture. Decide whether ShapeChange must also be
   baseline-relative and continuity-qualified.
3. Near-contact is currently normalized pair-hand distance <=1.0. That
   threshold is not yet calibrated from hand-contact annotations/c5 hard
   negatives, so it is a provisional geometry implementation rather than a
   validated shared-zone definition.
4. Hand landmarks may disappear for a few frames and reappear. Missingness is
   unknown evidence, not c5 evidence and not permission to switch to the other
   hand. Keep left/right hand streams separate; bridge a short observation gap
   only for identity/elapsed-time velocity, never as behavioral persistence.
   A return after a gap must use the same-hand baseline and must not fill
   missing shape or position with zero.

## Accepted hand-gap handling

For the next diagnostic, each actor has two independent hand streams. A
candidate hand `h` keeps its identity through missing frames using the actor's
stable left/right stream and pose-wrist continuity when available. During a
gap, `V_h=unknown`, not false; no Raise, ShapeChange, OwnSide, or SharedZone
claim is emitted. When observations resume, displacement and speed use the
elapsed time since the last valid observation. The actor may still receive a
class from one qualified resumed frame; no N-frame behavior persistence is
introduced.

## Hand-jitter warning and c1/c4 firewall

The user reports visible frame-to-frame hand landmark jitter while the real
hand is static. This is consistent with the context audit that recorded a
wrist discontinuity/change requiring hand-point continuity and bbox/crop QA.
Raw hand motion must therefore never be treated as a c7 signal by itself.

The next implementation must retain raw landmarks for audit but derive a
train-c5 jitter envelope per hand. A candidate hand displacement/speed is
usable only when its motion exceeds that envelope and is coherent across the
same hand's wrist plus valid fingertip/shape evidence. If quality or
coherence is insufficient, emit `unknown` evidence, not c7 and not a c5
motion negative. Filtering may suppress jitter, but filtered coordinates must
not manufacture an action persistence requirement.

For any future c1/c4-inclusive experiment, downward/retracting pose-arm cues
remain a separate branch. A hand-jitter-only frame cannot promote c7, and c7
must not outrank a qualified c1/c4 branch merely because `p7` is high. Current
c1/c4 metrics are not changed by this note; the firewall is a future design
constraint until those classes are reintroduced with actor-level evidence.

## Audit artifact

`data/processed/behavior_actor_only_c2_c3_c7_q7_pair_actor_score_v1/c7_q7_failed_actor_feature_scores.csv`
contains one row per failed c7-related actor decision and `mean/q95/max` for
every c7 cue, qualified-frame count, S7, and max p7-p5 before/after Q7.
