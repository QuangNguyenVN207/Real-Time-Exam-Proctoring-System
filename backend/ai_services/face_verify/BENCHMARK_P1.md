# Face verification P1

## Reproducible protocol

The benchmark consumes the actor manifest supplied outside the repository. A
row defines an enrollment/test source and the actors occupying the fixed left
and right seats. Notes are parsed only as actor-selection metadata.

```powershell
.\.venv-face\Scripts\python.exe -u -m backend.ai_services.face_verify.benchmark_video `
  --manifest "D:\minhphu\ML lab\final_project\face_test_data.csv" `
  --output-dir "data\face_benchmark_p1" `
  --model-root "data\insightface" `
  --sample-fps 2.0 `
  --max-templates 12 `
  --min-template-gap-s 0.75 `
  --det-size 640
```

- SCRFD-10G `det_10g.onnx` detects faces; ArcFace R50
  `w600k_r50.onnx` extracts
  512-dimensional L2-normalized embeddings.
- Videos are decoded chronologically and sampled at 2 FPS. Only the current
  frame is used, matching causal real-time inference.
- At most 12 quality-ranked enrollment templates are retained per actor with a
  minimum 0.75 s gap. Their normalized centroid is the deployed gallery entry.
- A query is accepted only when cosine top-1 is at least `0.32` and the gap to
  top-2 is at least `0.08`.
- Primary evaluation is actor-video level. Frame-level output is diagnostic.

## Result snapshot (2026-09-13)

- 12 enrolled identities, 13 known test actor-video groups, and 4 unknown test
  actor-video groups.
- 704 detected faces from 707 sampled actor-frame opportunities: 99.58%
  detection coverage.
- Selected rule (`0.32`, margin `0.08`): 13/13 known IDs, 4/4 unknown rejections,
  and 17/17 overall actor-video decisions. Diagnostic frame accuracy is 97.30%.

## Threshold and margin selection

The completed module uses the same pretrained InsightFace detector/recognizer,
multi-frame quality-ranked enrollment centroids, an enforced runner-up margin,
and assigned-ID verification that requires the expected identity to be top-1.

The complete experiment is implemented in `benchmark_video.py`. Function
`select_policy()` evaluates every combination of cosine threshold `0.20..0.75`
at step `0.01` and runner-up margin `0.00..0.15` at step `0.01`. Selection uses
actor-level known/unknown harmonic mean, open-set accuracy, and macro-F1. If a
plateau ties on all primary metrics, the threshold is chosen at the plateau
midpoint and the margin closest to the predeclared `0.08` is retained. The full
grid is exported as `threshold_margin_sweep.csv` and visualised in
`fig03_threshold_sweep.png`. `fig08_threshold_margin_heatmap.png` shows the
complete two-dimensional grid. On this calibration set, 400 combinations tie
on all primary actor metrics; at margin `0.08`, thresholds `0.20..0.44` all
belong to the optimum plateau. Therefore `0.32/0.08` must be described as the
selected stable operating point inside the best observed region, not as a
unique mathematical optimum.

The selected point is calibrated on the first 40% of each test actor-video stream and
evaluated on the remaining 60%. Because both partitions come from the same
recordings, it is an exploratory operating point, not proof of generalisation
to unseen cameras or capture sessions. The output includes Wilson 95% intervals
to make the small evaluation sample explicit.

## External pretrained-model evidence

The local sample is too small to establish broad population-level accuracy.
It is therefore reported separately from the upstream benchmark evidence. The
official InsightFace model zoo identifies `buffalo_l` as SCRFD-10GF plus a
ResNet-50 recognizer trained on WebFace600K, and reports recognition accuracy of
99.83% on LFW, 99.33% on CFP-FP, 98.23% on AgeDB-30, and 97.25% on IJB-C(E4).
These figures support the choice of pretrained backbone, but they do not replace
the local actor-level open-set test because capture conditions and protocols are
different.

- InsightFace model zoo: https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md
- ArcFace paper: https://arxiv.org/abs/1801.07698
- SCRFD paper: https://arxiv.org/abs/2105.04714

## Outputs and deployment

`data/face_benchmark_p1/` contains auditable CSV/JSON files, seven report-ready
PNG figures, and `gallery_embeddings.npz`. To use the multi-frame gallery in the
application, place that file at:

```text
data/student_faces/gallery_embeddings.npz
```

`FaceVerifier` loads this centroid gallery automatically. If it is absent, the
module remains backward-compatible and enrolls one face from each image in
`data/student_faces/`. The generated gallery contains biometric embeddings and
is intentionally excluded from Git by the repository's `data/` ignore rule.

The pretrained InsightFace model files are licensed by their publisher for
non-commercial research. The packaged P1 artifact is intended for academic
evaluation; redistribution or commercial use must follow the upstream terms.
