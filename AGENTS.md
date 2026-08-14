# Project behavior-classification contract

This instruction is mandatory for every task in this repository.

DO NOT ASSUME ANYTHING, VIEW EVERYTHING WITH SKEPTICISM

## Ultimate objective

Classify behavior of each actor. Never classify behavior of a whole video.

## Actor-only enforcement

- Remove video-level metrics, video-level coverage, video-level F1, video-level episode counts, and video-level label assumptions from analysis, artifacts, reports, and acceptance criteria.
- input is live video feed. In training, future frame must be masked to reflect this, every metric or feature must be computed using previous frames
- Every truth label, feature aggregation, prediction, threshold audit, confusion matrix, precision, recall, F1, coverage, and false-alert metric must be actor-level, keyed at minimum by `(video, actor_id)`. Temporal rows may support an actor decision but may not be the primary metric.
- `action_actor_ids` is the complete set of actors performing manifest `class_code`. Listed actors receive that class; unlisted actors receive c5.
- `interaction_pairs` describes source/peer relation and does not override `action_actor_ids` truth.
- `action_start_s/action_end_s` are ground-truth onset/end. Scan the full video; never crop model input to this interval.
- No persistence requirement. One qualified positive frame is enough to assign an actor an action class. Positive evidence may be intermittent.
- Actor state starts temporary c5. A later positive class replaces the current actor class only when its score is higher. Keep the highest-scoring actor decision; later negative frames do not revert it. Train c5 score experimentally; never assume zero.
- c1/c4/c6, c7 are excluded from current target. Target classes are c2/c3/c7. Do not mix an actor whose truth is c2/c3 into c5.
- Predicted contiguous runs are not ground-truth actor episodes.



## Protected base branch

- `codex/development-benchmark-v3` is the base branch.
- Do not commit, push, force-push, merge, or otherwise modify the base branch without explicit user approval.
- When the user requests restoration of the base branch, restore files, folders, and code to match `codex/development-benchmark-v3` exactly.
- Restoration may include deleting generated outputs and undoing code changes when necessary.
- Before any restoration that may remove user data, inspect and report the exact scope of changes.
