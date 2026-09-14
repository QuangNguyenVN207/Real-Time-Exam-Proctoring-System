# Pose–gaze causal runtime package

The deployment source of truth is
`tmp/causal_8fps_stage6_mixed_084699_final_20260827`. The runtime loads exactly
three final XGBoost models: C2 exchange (`65` ordered features), C3 looking
(`90`), and `suspicious_activity` (`50`). Fold models and OOF files remain in
the archive because the strict Stage 6 manifest covers all 41 bundle files.

The package also contains the exact `yolov8n.pt` person detector and MediaPipe
`holistic_landmarker.task`. Therefore a restored package does not silently
download a different landmark model.

`backend/ai_services/face_verify/identity_guard.py` is included unchanged as
the lightweight integration bridge imported by `PoseGazePaperPipeline`. The
face model and face runtime are not duplicated or modified by this package.

## Locked causal configuration

- Sampling: 8 FPS. Application input is timestamp-sampled before inference.
- Valid baseline frames: 4; head-turn baseline frames: 8.
- Rolling window: 24 sampled frames.
- Maximum derivative gap: 450 ms.
- C2 threshold: `0.31567803025245667`.
- C3 threshold: `0.22652508318424225`.
- Suspicious-activity threshold: `0.6603246927261353`.
- C2 requires an explicit actor pair such as
  `student_01:student_02`; the pair relationship does not change actor truth.

All gate thresholds are machine-readable in `POSE_GAZE_RELEASE.json` and
`calibration.json`. The bundle loader checks hashes, ordered schemas, temporal
policy, calibration, and model identities before camera processing starts.

## Restore and verify

Extract the ZIP at the repository root, then use Python 3.12:

```powershell
python -m venv .venv-pose-gaze
.venv-pose-gaze\Scripts\python.exe -m pip install -r backend\ai_services\pose_gaze\requirements-runtime.txt
.venv-pose-gaze\Scripts\python.exe -m backend.ai_services.pose_gaze.verify_release
```

On Linux or macOS, use `.venv-pose-gaze/bin/python`. The verified portable
baseline runs XGBoost on CPU. The same `.ubj` model files may use a compatible
CUDA build, but the release never substitutes another model.

Build the deterministic archive with:

```powershell
.venv-pose-gaze\Scripts\python.exe -m backend.ai_services.pose_gaze.package_release
```

The external `.manifest.json` and the in-archive
`POSE_GAZE_PACKAGE_MANIFEST.json` record every included file and SHA-256.
