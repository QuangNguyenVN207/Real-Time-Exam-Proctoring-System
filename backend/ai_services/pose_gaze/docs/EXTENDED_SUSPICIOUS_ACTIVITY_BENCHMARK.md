# Extended causal benchmark with `suspicious_activity`

This is an extended profile. It does not replace the official benchmark.

## Label contract

The extended actor-level labels are:

```text
[suspicious_activity, c2, c3, c5, c7]
```

Pose-only C1 and C4 actors map to `suspicious_activity`. Pose alone cannot
reliably distinguish phone from paper; object fusion is required before
exposing C1 versus C4 again.

The official benchmark remains the original-label causal C2/C3/C5 replay:

```text
[c2, c3, c5]
```

See `OFFICIAL_CAUSAL_BENCHMARK.md`. Its locked macro-F1 remains `0.8730158730`.

## Extended artifact

- Artifact: `tmp/behavior_actor_end_to_end_causal_pose_only_20260813`
- Protocol: actor-level causal live-feed specialist replay
- Future frames: forbidden
- Current extended macro-F1: `0.5719658120`
- This result uses raw cross-specialist score competition and is not a
  replacement for the official C2/C3/C5 metric.

## Reproduction

```powershell
\.venv\Scripts\python.exe -m backend.ai_services.pose_gaze.pose_gaze.holistic.feature_csv.benchmark_end_to_end_causal
```

This command writes the extended profile only. It cannot overwrite the
official artifact unless an explicit output directory is supplied.
