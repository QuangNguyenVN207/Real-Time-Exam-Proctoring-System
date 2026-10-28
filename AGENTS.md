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

## Project orientation for AI agents (read before exploring)

Start at `README.md` section 0 ("Đọc nhanh"): it holds the architecture map, the numbers already measured
on the reference machine, the known traps, and the list of files that are gitignored but required at
runtime. Sections 2-11 are the detail. **Do not re-measure what section 0 / section 8 already states** —
reuse the figures and say which machine they came from (Intel Core Ultra 7 155H + iGPU Arc, Ubuntu,
Python 3.12.3). NVIDIA / Windows numbers are **not** measured yet, so treat any CUDA claim as unverified.

- **One Streamlit process, one entry point.** `app_webrtc.py` is the whole app; `app_webrtc_win.py` is a
  12-line `runpy` shim kept only so old launch commands still work. Do not re-split them.
- **Shared detection code.** Per-frame logic lives in `backend/ai_services/pipeline/` (pure functions: no
  threads, no `st.*`). It is used by the realtime threads, the in-UI offline block and `run_offline.py`.
  Fix it once there, never inside an app file.
- **Device selection is keyed on hardware, not OS.** `select_device().kind` (`cuda` / `intel` / `cpu`)
  picks the model path and devices, so WSL + NVIDIA gets the CUDA profile; `PROCTOR_DEVICE` overrides it.
  Only the microphone worker is OS-keyed (`sounddevice` on Windows, `PyAudio` on Linux/WSL).
- **Offline mirrors realtime.** `run_offline.py <video>` writes `annotated.mp4` (burned overlays + warning
  panel), `events.jsonl`, `log.txt`, `report.json`. `report.json` keys evidence by `(video, actor_id)` in
  `actors`, including actors whose final class is `c5` — that satisfies the actor-level rule above.
- **Deliberate trade-offs, not bugs** (ask before changing): the duplicated gaze stack (README section 9
  item 1), `alerts_by_module` counting paper `status="info"` frames (item 11), the 4s UI autorefresh
  interval (section 11.4, measured 3.3x inference slowdown at 1.2s), and `mp4v` output for CLI with H.264
  generated only for browser playback.
- **Camera vs microphone in WSL.** WebRTC frames come from the browser, so no `/dev/video0` is needed in
  WSL; the microphone is captured server-side and is the fragile part there. The offline flow needs no
  camera or microphone — prefer it for verifying detection behaviour on a new machine.
