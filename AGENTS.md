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
- **The UI offline job runs as a subprocess.** `OfflineJob` spawns `run_offline.py --progress-jsonl`, parses
  `@@PROGRESS@@` lines and cancels with SIGTERM. Do not move it back into the Streamlit process: the app
  already holds ~3GB of realtime models, and a second in-process model set exhausted 15GB RAM and triggered
  the kernel OOM killer (2026-10-28). The child sets `oom_score_adj=500` so it dies before the UI.
- **TTL overlay is display-only.** `--ttl 0.5` and `1.5` measured the same wall time (131.1s vs 130.0s) and
  overlay count; it never explains RAM or speed differences. RAM comes from model sets and modules enabled.
- **Two test clips, different behaviour.** `data/IMG_7817.MOV` produces no behaviour alerts (both actors end
  `c5`); `data/WIN_20260802_12_05_12_Pro.mp4` does (28 `pose_gaze` alerts for `student_01`, 532 frames).
- **Exactly one behaviour stack.** `pipeline/gaze_scan.py` (`GazeStack`) is the only tracking + holistic +
  causal classifier in the runtime; skeleton drawing reuses its `latest_packet`/`latest_results` via
  `GazeStack.draw_skeleton()`. `PoseGazeDetector` is legacy — it survives only in `app_ubuntu.py`,
  `app_win.py` and `backend/main.py`, which are not entry points. Do not reintroduce a second stack.
- **Measure, don't guess.** `pipeline/metrics.py` (`METRICS.time(stage)` / `.drop(stage)` / `.snapshot()`)
  is wired into the app threads and rendered in the "🩺 Sức khoẻ module" panel. For camera-free A/B, run
  `scripts/replay_realtime.py <video> [--fast]` — it rebuilds the same 4 threads and `maxsize=2` queues.
  Reuse its output and README section 8.2 rather than re-measuring.
- **Behaviour scores are not reproducible.** Two runs of identical code give different `c2`/`c3`/
  `suspicious` scores (README section 0). A/B by decisions/alert counts and latency only; never by score.
- **`face_scan.scan_stranger_with_masking` detects once per frame.** The old masked re-detect path was
  removed (2.68x faster, fewer false alerts on the test clip); its name is kept for call sites. Do not
  restore masking without measuring both alert counts and per-frame decisions.
- **Sidebar log throttling only.** `LOG_COOLDOWN = 2s` collapses duplicate log lines on the UI; overlays
  are still refreshed and `events.jsonl` / console still record every detection. Never count alerts from
  the sidebar — use `report.json` / `events.jsonl`.
- **Deliberate trade-offs, not bugs** (ask before changing): `alerts_by_module` counting paper
  `status="info"` frames (README section 9 item 11), the 4s UI autorefresh interval (section 11.4,
  measured 3.3x inference slowdown at 1.2s), and `mp4v` output for CLI with H.264 generated only for
  browser playback.
- **Camera vs microphone in WSL.** WebRTC frames come from the browser, so no `/dev/video0` is needed in
  WSL; the microphone is captured server-side and is the fragile part there. The offline flow needs no
  camera or microphone — prefer it for verifying detection behaviour on a new machine.
