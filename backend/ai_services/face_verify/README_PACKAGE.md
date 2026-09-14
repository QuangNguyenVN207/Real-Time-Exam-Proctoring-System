# Face P1 package usage

This directory is the locked academic hand-off for Face P1. It includes the
runtime source, exact configuration, pretrained ONNX files, the local centroid
gallery, tests, benchmark outputs, figures, and file-level checksums. Raw input
videos are not duplicated in the package.

## Restore into the project

From the release root, copy:

```text
runtime_data/insightface/models/buffalo_l/*
    -> data/insightface/models/buffalo_l/
runtime_data/student_faces/gallery_embeddings.npz
    -> data/student_faces/gallery_embeddings.npz
```

Keep the `backend/` tree at the repository root. The application then uses the
locked values from `backend/core/config.py` automatically.

## Isolated smoke test

Create an environment and install the face-only dependency lock:

```powershell
python -m venv .venv-face
.\.venv-face\Scripts\python.exe -m pip install -r backend\ai_services\face_verify\requirements-benchmark.txt
.\.venv-face\Scripts\python.exe -m unittest `
  backend.ai_services.face_verify.test_face_verify_logic `
  backend.ai_services.face_verify.test_benchmark_video `
  backend.ai_services.face_verify.test_identity_guard
```

Load the packaged models and gallery without moving them:

```python
from backend.ai_services.face_verify.face_verify import FaceVerifier

verifier = FaceVerifier(
    db_path="runtime_data/student_faces",
    model_root="runtime_data/insightface",
)
```

Expected locked state: 12 gallery identities, embedding matrix shape `(12,
512)`, cosine threshold `0.32`, and runner-up margin `0.08`.

Review `FACE_P1_RELEASE.json` for machine-readable configuration,
`BENCHMARK_P1.md` for development history and interpretation, and
`benchmark_results/report_snippet.md` plus the PNG files for report material.

The gallery contains biometric embeddings. Keep it local and limit access.
InsightFace pretrained model files are intended here for non-commercial
academic research under the publisher's model terms.
