# Object detection runtime package

This release always uses `weights/best (1).pt`. It does not contain or select
an OpenVINO export. The checkpoint SHA-256 is locked in `OBJECT_RELEASE.json`,
and the verifier rejects a renamed, missing, altered, or class-incompatible
checkpoint.

The locked class order is `smartwatch`, `earphone`, `cheat_sheet`,
`smartphone`, `calculator`. Runtime alert thresholds follow the detailed slide:
default `0.50`, earphone `0.55`, smartphone `0.55`, and paper `0.20`.
Non-paper alerts use three matching detections in a five-inference window.

## Restore and verify

Extract the ZIP at the repository root, then use Python 3.12:

```powershell
python -m venv .venv-object
.venv-object\Scripts\python.exe -m pip install -r backend\ai_services\object_detect\requirements-runtime.txt
.venv-object\Scripts\python.exe -m backend.ai_services.object_detect.verify_release
```

On Linux or macOS, replace the final two commands with the corresponding
`.venv-object/bin/python` path. Device selection is CUDA first, then Apple MPS,
then CPU. Every path uses the same `.pt` checkpoint; Intel GPUs fall back to
CPU rather than silently selecting an incomplete OpenVINO export.

Build the deterministic archive with:

```powershell
.venv-object\Scripts\python.exe -m backend.ai_services.object_detect.package_release
```

The external `.manifest.json` and the in-archive
`OBJECT_PACKAGE_MANIFEST.json` record every included file and SHA-256.
