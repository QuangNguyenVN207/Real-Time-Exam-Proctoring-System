# Project behavior-classification contract

This instruction is mandatory for every task in this repository.

**DO NOT ASSUME ANYTHING, VIEW EVERYTHING WITH SKEPTICISM AND ASK ME EVERYTIME**

## Ultimate objective

Classify behavior of each actor. Never classify behavior of a whole video.
Real time classification with hardware limitation in mind

## Actor-only enforcement
- input is live video feed. future frame must be maske, every metric or feature must be computed using previous frames
- Every truth label, feature aggregation, prediction, threshold audit, confusion matrix, precision, recall, F1, coverage, and false-alert metric must be actor-level, keyed at minimum by `(video, actor_id)`. Temporal rows may support an actor decision but may not be the primary metric. **can be ignore for new pipeline**
- `action_actor_ids` is the complete set of actors performing manifest `class_code`. Listed actors receive that class; unlisted actors receive c5.
- `interaction_pairs` describes source/peer relation and does not override `action_actor_ids` truth.
- `action_start_s/action_end_s` are ground-truth onset/end. Scan the full video; never crop model input to this interval.
- No persistence requirement. One qualified positive frame is enough to assign an actor an action class. Positive evidence may be intermittent. **can be ignore for new pipeline**
- Actor state starts temporary c5. A later positive class replaces the current actor class only when its score is higher. Keep the highest-scoring actor decision; later negative frames do not revert it. Train c5 score experimentally; never assume zero. **for new pipieline, this can be ignored**
- Target classes are c1/c4/c2/c3/c5. Do not mix an actor whose truth is c2/c3 into c5. c1 and c4 is combined into superclass 'suscpicous_activity', this superclass is combined with YOLO (weight best.pt or last.pt) to divide into c1, c4. Note: YOLO doesn't need to detect continuously the cheating object, one frame with high enough confidence is sufficent
- Predicted contiguous runs are not ground-truth actor episodes.
- **for new pipeline**: landmark must be extracted in that frame like real time, not the entire json landmark
- **Dataset**: approximate: 90 actors with 120+ videos
- **for new pipeline**: use morden model (call API if needed) 



## Protected base branch
- `codex/yolo-detection` is the base branch.
- Do not commit, push, force-push, merge, or otherwise modify the base branch without explicit user approval.
- When the user requests restoration of the base branch, restore files, folders, and code to match `codex/yolo-detection` exactly.
- Restoration may include deleting generated outputs and undoing code changes when necessary.
- Before any restoration that may remove user data, inspect and report the exact scope of changes.
- **for new pipeline**: Work on branch `new-test-pipieline` 