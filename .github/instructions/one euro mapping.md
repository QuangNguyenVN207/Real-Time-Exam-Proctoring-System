## Plan: One Euro Face World Mapping

Thêm One Euro smoothing cho face global coordinates, rồi map face sang pseudo-world 2D bằng pose anchors. Dùng Kabsch 2D làm transform chính vì giữ rotation + scale + translation, dùng OpenCV affine 2D với 4 điểm như nhánh benchmark/diagnostic, không dùng đồng thời hai transform trong feature production.

**Steps**
1. Xác nhận contract dữ liệu hiện tại trong `pose_gaze/holistic/landmark/landmarks.py`, `TrackHolisticResult`, `LandmarkPoint`, JSON writer, và `holistic/feature_csv/feature_csv.py`. Giữ format cũ đọc được; tăng schema version chỉ khi thêm field mới.
2. Tạo module nhỏ trong code root `pose_gaze/holistic/landmark/face_mapping.py`:
   - `OneEuroFilter2D` hoặc filter tương đương, state theo `track_id` và landmark index, dùng timestamp milliseconds.
   - Kabsch 2D từ 4 pose anchors `(11, 12, 23, 24)` khi đủ confidence; tính centroid, rotation, scale, translation, reject khoảng cách quá nhỏ và NaN/Infinity.
   - OpenCV affine 2D 4-point helper dùng `cv2.estimateAffine2D` hoặc `cv2.getAffineTransform` trên 3 điểm + đánh giá điểm thứ 4; chỉ benchmark/diagnostic, vì affine có thể thêm shear.
   - Face pseudo-world mapper: shoulder pair primary, hip pair fallback; map face global pixel coordinates về frame-normalized/pixel pose coordinates trước transform; output `world_x`, `world_y`, `world_valid`, `anchor_source`.
3. Mở rộng `LandmarkPoint` với `world_x`, `world_y` và chỉ serialize fields này khi có giá trị; giữ `x/y/frame_x/frame_y` nguyên nghĩa normalized/frame pixel. Không gọi pseudo-world là MediaPipe true world.
4. Tích hợp trong `process_track`: pose anchors lấy từ pose normalized đã bỏ letterbox và pose world; map observed face sau khi đưa từ ROI về global; áp dụng One Euro trên global face points trước pseudo-world mapping; held face remap theo crop hiện tại, không cập nhật world khi `face_predicted=true` nếu anchor/face không hợp lệ; reset filter/mapping state khi track bị discard hoặc person đổi.
5. Cập nhật `feature_csv.py`: quyết định schema rõ ràng. Giữ feature face `x/y` tương thích; thêm nhóm `face_world` chỉ khi downstream model cần pseudo-world. Tăng `CSV_SCHEMA_VERSION`, cập nhật `POINT_FIELDS`/`LANDMARK_GROUPS`, `CSV_FIELDNAMES`, quality masks; không regenerate artifacts trong cùng change nếu chưa có yêu cầu train lại.
6. Thêm test trong `pose_gaze/holistic/tests/test_landmarks` hoặc module test kế bên:
   - One Euro giảm jitter khi đứng yên, phản ứng nhanh hơn EMA khi có bước dịch lớn.
   - Kabsch khôi phục known rotation/scale/translation với 4 anchors.
   - affine 4-point helper reject thiếu/degenerate anchors và không phát NaN.
   - shoulder mapping được ưu tiên; hip fallback khi confidence thấp; null fields khi cả hai thiếu.
   - face ROI đổi nhưng global face giữ ổn định; hold remap đúng; `face_predicted` không bị tính như observed face.
   - JSON/CSV round-trip không có NaN/Infinity; field cũ vẫn đọc được.
7. Chạy validation từ `backend/ai_services/pose_gaze` với đúng `PYTHONPATH`: `python -m py_compile ...`; pytest focused với plugin autoload tắt; chạy validator JSON; chạy A/B trên một clip ngắn trước khi full c3/c6.
8. Chạy full c3/c6 sau khi test pass, tính coverage face/mouth theo frame-track và jitter RMS global. So sánh baseline EMA với One Euro; chỉ retrain/evaluate F1 sau khi CSV train/val/test được regenerate cùng mapping và smoothing.

**Relevant files**
- `C:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\holistic\landmark\landmarks.py` — integration, state cleanup, face global remap, output construction.
- `C:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\holistic\feature_csv\feature_csv.py` — model feature schema and CSV flattening.
- `C:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\holistic\tests\test_landmarks\test_landmarks.py` — existing extraction/schema tests.
- `C:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\holistic\test_media\validate_landmarks.py` — structural JSON validation.
- `C:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\holistic\test_media\test_media.py` — JSON output integration path.

**Verification**
1. Unit tests assert Kabsch residual, shoulder/hip selection, One Euro response, and reset behavior.
2. `python -m py_compile` on changed modules.
3. Focused pytest with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and package root configured; fix package invocation before interpreting test failures.
4. Validate representative JSON: coordinate ranges, `world_valid`, no NaN/Infinity, held face flags.
5. Compare c3/c6 baseline vs One Euro: face coverage, mouth coverage, face jitter RMS, fallback count, and downstream F1 only after feature regeneration.

**Decisions**
- Pseudo-world is 2D derived representation, not MediaPipe true face world landmarks.
- Kabsch is production transform. OpenCV affine 4-point is benchmark/diagnostic unless experiments prove it improves validation error without instability.
- Shoulder anchors `(11, 12)` primary; hip anchors `(23, 24)` fallback. Confidence threshold uses `DEFAULT_HOLISTIC_CONFIDENCE`; reject missing, low-confidence, degenerate, or non-finite values.
- One Euro applies after ROI-to-global conversion; no independent smoothing of ROI-normalized face coordinates.
- Scope follows repository memory: modify only files under `backend/ai_services/pose_gaze/pose_gaze`; do not touch outer package or existing generated artifacts unless explicitly required.

**Further Considerations**
1. Confirm production feature contract before implementation: add `face_world_*` columns now, or keep pseudo-world JSON-only until F1 experiment. Recommendation: JSON/internal first, CSV columns in separate retraining change.
2. Confirm affine role: use for comparison only, or allow runtime fallback when Kabsch residual exceeds threshold. Recommendation: comparison first; runtime fallback can hide anchor/data problems.

**Locked experiment protocol**
- `face_predicted=true` rows are excluded absolutely from training and validation feature rows. No imputation from held face. Keep them only in audit JSON and report excluded counts/duration per video.
- Keep a separate `face_observed_mask` for valid fresh face observations. `face_valid` remains observation status; it must not be inferred from displayed held points.
- Four feature variants: `baseline`, `baseline + one_euro_face`, `baseline + pseudo_world`, `baseline + one_euro_face + pseudo_world`.
- One Euro is independently switchable. It is not silently enabled inside pseudo-world mapping.
- Mapping comparison: Kabsch 2D and OpenCV affine 2D. Kabsch is production mapping. Affine is benchmark only; do not train separate affine models unless benchmark proves clear benefit and a new experiment is explicitly recorded. Production model runs: 4.
- Production runs: `baseline`, `baseline + one_euro_face`, `baseline + pseudo_world_kabsch`, `baseline + one_euro_face + pseudo_world_kabsch`.
- Affine benchmark runs only on validation mapping quality: residual, finite ratio, anchor-valid ratio, jitter, and reacquisition. Do not mix affine output into production training CSV.
- Split by `split_group`/recording/session/video before row generation. Never random-split frames or windows from same video across train/val/test. Fit scaler/feature selection on train only.
- Primary model metrics: window-level macro F1, balanced accuracy, and multiclass ROC-AUC OvR macro. Report weighted F1 and micro F1 as secondary, not alone.
- Imbalance metrics: per-class precision/recall/F1, support, confusion matrix, macro PR-AUC OvR, and Matthews correlation coefficient. PR-AUC is required for rare classes; ROC-AUC alone can look good under imbalance.
- Video-level metrics: aggregate window probabilities by video using mean probability and majority vote; report macro F1, balanced accuracy, macro ROC-AUC where class support permits, per-class recall, and video accuracy. Keep window-level and video-level results separate.
- Temporal behavior metrics: event-level detection recall/precision/F1, onset delay, false alarms per minute, and prediction stability (number of label switches per minute). Compute only from timestamped windows and annotation intervals; do not invent event metrics for videos without reliable action intervals.
- Landmark quality metrics: observed face coverage, mouth coverage, excluded predicted-frame ratio, reacquisition gap, global face jitter RMS normalized by face/shoulder scale, anchor-valid ratio, mapping residual, and fraction of finite pseudo-world points. Report per video and macro across videos; duration-weighted aggregate alone is insufficient.
- A/B pass rule: choose by validation macro F1 first, then macro PR-AUC/balanced accuracy; reject any candidate with worse rare-class recall, excessive false alarms per minute, or materially worse jitter/reacquisition. Test split used once after selection.
- Report confidence intervals by video bootstrap, not frame bootstrap. Use paired per-video deltas against baseline. Preserve random seed, model metadata, feature schema, mapping name, One Euro parameters, excluded rows, and video list for every run.

**Execution gates**
1. Implement One Euro, Kabsch, pseudo-world, `face_observed_mask`, and predicted-row filter.
2. Add unit tests. Require `py_compile` and focused tests pass.
3. Run short-video smoke extraction. Validate JSON schema, finite coordinates, predicted-row exclusion, and track reset.
4. Run full video extraction for canonical manifest. Keep raw JSON audit separate from filtered training rows.
5. Build frame-track rows, remove every `face_predicted=true` row, then build temporal windows. Never impute held face.
6. Verify split leakage by video/session/`split_group`. Fail run on overlap.
7. Generate four production feature sets with identical rows, labels, splits, seed, scaler policy, and window policy.
8. Train four models. Fit scaler and feature selection on train only.
9. Evaluate validation with model and landmark metrics. Select winner by macro F1, then macro PR-AUC and balanced accuracy.
10. Run affine benchmark on same validation videos. Keep result diagnostic only.
11. Run selected model once on test. Report window-level, video-level, temporal, and landmark metrics.
12. Export model metadata: feature schema, variant, mapping, One Euro config, split groups, excluded counts, seed, and source video list.

**Run count**
- Train: `4` production models.
- Affine: benchmark only, no production model.
- Test: `1` selected model, after validation lock.

**New-chat handoff**

New chat starts implementation immediately from this section. Do not redesign protocol or ask run-count questions unless code inspection reveals contradiction.

**Scope**
- Modify only `backend/ai_services/pose_gaze/pose_gaze/**`.
- Do not modify generated JSON, CSV, model weights, or outer `backend/ai_services/pose_gaze` package during implementation.
- Keep `face_predicted=true` rows in audit JSON only. Exclude them absolutely from train, validation, scaler fitting, window building, and metrics.
- Production mapping: Kabsch 2D. OpenCV affine 2D: validation benchmark only.

**Implementation order**
1. Inspect current `landmarks.py`, `LandmarkPoint`, `TrackHolisticResult`, `feature_csv.py`, and batch dataset writer.
2. Add `holistic/landmark/face_mapping.py` with One Euro, Kabsch 2D, affine benchmark, anchor selection, and finite-value guards.
3. Add face pseudo-world fields without breaking old JSON readers. Preserve `x/y/frame_x/frame_y` meanings.
4. Integrate observed-face global coordinates, One Euro, Kabsch pseudo-world mapping, `face_observed_mask`, and track reset.
5. Add predicted-row exclusion at dataset row/window boundary, not only in model code.
6. Add focused unit tests before full extraction.
7. Run `py_compile`, focused pytest, JSON validator, and short-video smoke test.
8. Generate full canonical JSON only after gates pass.
9. Generate four production feature sets and train four models.
10. Run affine benchmark on identical validation inputs.

**First action in new chat**

Read this plan plus `.github/instructions/copilot-instructions.md` and `.github/instructions/context.md`. Then inspect `landmarks.py` and `feature_csv.py`. State one local hypothesis and one focused test. Apply first small edit immediately; run focused validation before more exploration.

**Required commands**

Run from repository root:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:PYTHONPATH = "backend/ai_services/pose_gaze;."
python -m py_compile backend/ai_services/pose_gaze/pose_gaze/holistic/landmark/landmarks.py
python -m pytest -q backend/ai_services/pose_gaze/pose_gaze/holistic/tests/test_landmarks backend/ai_services/pose_gaze/pose_gaze/holistic/tests/test_feature_csv
```

If package collection fails, report exact import error and fix only package invocation/path. Do not interpret collection failure as feature failure.

**Done criteria**

- Four production variants reproducibly train from same filtered rows and split groups.
- No `face_predicted=true` row enters train, validation, scaler, windows, or metrics.
- Kabsch and affine benchmark produce finite diagnostics.
- Validation selects one model using macro F1, macro PR-AUC, and balanced accuracy.
- Test runs once after selection.
- Report includes per-video and macro metrics, excluded predicted rows, face coverage, mouth coverage, reacquisition gap, jitter RMS, anchor validity, and mapping residual.
