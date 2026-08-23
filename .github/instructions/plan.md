## Plan: Pose Gaze R4 Holistic Research

Nghiên cứu và chốt thiết kế cho pipeline Holistic duy nhất `backend/ai_services/pose_gaze/pose_gaze`, gồm hai entrypoint cùng dùng chung contract: webcam realtime (`main`) và batch Holistic (`holistic/batch_dataset`). Bám sát paper R4 2017, đối chiếu paper Exploring skeleton ML, rồi nâng cấp tracking, schema CSV landmark frame-track, CSV temporal-window feature cho XGBoost, và manifest tương thích ngược. Stage 1/2 bị loại khỏi pipeline; chỉ được nhắc như nguồn tham chiếu lịch sử nếu cần. User đã cho phép và đã tạo `backend/ai_services/pose_gaze/context.md` sau khi chốt plan; implementation tracking vẫn pending.

**Steps**
1. Khóa phạm vi nghiên cứu: chỉ dùng `c:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze`, `c:\Real-Time-Exam-Proctoring-System\data\processed`, `c:\Real-Time-Exam-Proctoring-System\RESEARCH.md`, và 2 PDF đã chốt là `R4_CheatingVideoDescriptionBasedonSequencesofGestures_rev2 (2).pdf` và `Exploring_the_potential_of_skeleton_and_machine_le (5).pdf`. Bỏ toàn bộ folder ngoài phạm vi này.
2. Đối chiếu paper R4 2017 với pipeline Holistic hiện có trong `pose_gaze/pose_gaze`: xác định yêu cầu bắt buộc về chunk/window, hai chủ thể trong cùng cảnh, gesture sequence, feature tabular cho XGBoost, và nơi current pipeline đã có hoặc còn thiếu. Đây là bước chặn cho toàn bộ đề xuất tiếp theo.
3. Đối chiếu paper Exploring skeleton ML như paper phụ: trích các yêu cầu có giá trị trực tiếp cho feature engineering từ landmark, đặc biệt joint angles, bone relations, motion/velocity, baseline normalization, và xác định phần nào nên nhập vào pipeline mà không làm lệch mục tiêu R4. Bước này phụ thuộc bước 2 nhưng có thể tổng hợp song song trong cùng báo cáo nghiên cứu.
4. Kiểm tra pipeline Holistic thực tế đang dùng tại `c:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\holistic\landmark\landmarks.py` và `c:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\holistic\feature_csv\feature_csv.py`: chốt nhóm landmark nào đang được materialize, CSV metadata nào đang có, và vector feature nào runtime XGBoost đang tiêu thụ. Đây là chuẩn gốc để mọi đề xuất không đi chệch pipeline thật.
5. Đánh giá `c:\Real-Time-Exam-Proctoring-System\data\processed\stage0_manifest.csv`: xác nhận các cột hiện tại đã đủ cho export Holistic hay chưa; phân biệt rõ cột nào đủ cho extraction/labeling/splitting và cột nào còn thiếu cho directional pair reasoning, camera/view metadata, quality certainty, frame count. Vì người dùng chọn tương thích ngược, chỉ đề xuất thêm tối thiểu, không thay đổi hoặc phá vỡ cột hiện có.
6. Chốt layout CSV trong chính `pose_gaze/pose_gaze`:
   6.1. `frame_track.csv`: một hàng cho mỗi `frame_id + track_id`, chứa metadata thời gian/identity/quality và flattened Holistic landmarks theo `MODEL_FEATURE_COLUMNS` hoặc schema v2 tương thích.
   6.2. `window_features.csv`: một hàng cho mỗi temporal window và directional relation `source_student_id -> peer_student_id`, dùng làm input chính cho XGBoost theo R4.
   6.3. Giữ raw landmark arrays tùy chọn ở NPZ/parquet chỉ để debug/reprocessing; không thay thế hai CSV contract chính.
   6.4. Không dùng `windows.parquet`, `selected_frames.parquet`, hoặc Stage 1/2 artifact làm đầu vào bắt buộc của pipeline mới.
7. Chốt temporal-window contract cho XGBoost:
   7.1. Frame-level CSV là nguồn dữ liệu trung gian, không phải sample cuối cùng của R4 classifier.
   7.2. Window builder chạy trong `pose_gaze/pose_gaze`, nhận frame-track rows theo thứ tự timestamp/frame_id, tạo window cố định theo thời gian và lưu frame_count/valid_frame_count.
   7.3. Window features gồm head pose/gaze/body features hiện có, joint angles, hand velocity, bone motion, track continuity, quality ratios, và pairwise A→B/B→A features.
   7.4. Giữ c1 phone cheating và c4 cheat-sheet cheating tách biệt; giữ đủ 7 class c1–c7 trong label contract.
8. Thiết kế nâng cấp tracking runtime:
   8.1. Giữ nguyên `BoundingBox`, `PersonDetection`, `TrackedPerson`, `TrackPacket` làm handoff contract.
   8.2. Phase 1 giữ `IoUPersonTracker`; sửa duplicate suppression và association trước, ưu tiên identity continuity khi mất detection ngắn/che khuất/giao nhau, sau đó mới giảm bbox jitter và ổn định crop cho Holistic.
   8.3. Đưa foreground policy vào runtime: detector class/confidence gate, deduplicate, associate existing tracks trước, rank candidate mới bằng confidence + bbox area ratio + seat/ROI plausibility khi có, cap hai active student tracks.
   8.4. Giữ `TrackingManager` chịu trách nhiệm session, student assignment, persistence và remap; không đưa windowing/batch logic vào manager.
   8.5. Có thể mượn logic/threshold/quality ideas từ Stage 2, nhưng không coi `_select_front_tracks()` là runtime dependency; code chạy thật chỉ nằm trong `pose_gaze/pose_gaze/tracking`.
   8.6. Chỉ benchmark ByteTrack sau khi baseline IoU pass; nếu thêm, bọc sau cùng foreground policy, giữ contract và `max_tracks=2`, không bật mặc định.
   8.7. Không tăng `DEFAULT_PERSON_CONFIDENCE`, `min_area_ratio`, hoặc Stage 2 thresholds chỉ để che duplicate bug; mọi threshold mới phải có fixture/benchmark cho hai học sinh thật và outsider.
9. Bổ sung phase manifest và dataset ingest cho Batch Holistic:
   9.1. `stage0_manifest.csv` là input của `holistic/batch_dataset`, không phải input của webcam realtime.
   9.2. Giữ nguyên toàn bộ cột hiện tại: `clip_id`, `filename`, `recording_session`, `actor_ids`, `action_actor_ids`, `class_code`, `layout`, action interval, review, quality, split và exclusion flags.
   9.3. Bổ sung additive metadata khi chưa suy ra được từ media: `media_type`, `camera_view_id`, `source_fps` hoặc `fps_verified`, `annotation_confidence`, và `interaction_pairs` dạng JSON cho quan hệ source/peer.
   9.4. Không bắt buộc ghi tay `frame_count`, `frame_width`, `frame_height`, `actual_fps`; batch runner phải đọc media và ghi các giá trị đo được vào output metadata.
   9.5. Tạo normalized batch manifest nội bộ từ stage0 manifest và các video mới: chuẩn hóa đường dẫn media, parse JSON actor fields, kiểm tra media tồn tại, xác định FPS/frame count, giữ action interval theo timestamp.
   9.6. Dùng video gốc làm source ưu tiên cho 20 video mới. `data/processed/train`, `val`, `test` hiện chứa frame/window artifacts của Stage 1; không dùng trực tiếp trong pipeline mới.
   9.7. Nếu không còn video gốc, cho phép image-manifest fallback chỉ khi có `clip_id`, `source_frame_index`, `timestamp_ms` hoặc FPS, thứ tự frame ổn định, label, và split; không nhận một thư mục ảnh không có temporal metadata.
   9.8. Không trộn frame artifacts cũ với video mới trong cùng sample contract nếu chưa chứng minh cùng preprocessing, FPS, label semantics và split policy.
   9.9. Batch runner chuyển video-level row thành frame-track rows rồi thành temporal windows trong `pose_gaze/pose_gaze`; không dùng `windows.parquet`, `selected_frames.parquet`, hoặc Stage 1/2.
   9.10. Giữ split train/val/test theo clip/session/subject group, không random frame. Kiểm tra 20 video mới không làm một session hoặc student xuất hiện chéo train/test.
   9.11. Thêm validation: class chỉ thuộc c1-c7; đủ coverage class; c1/c4 không trộn; c5 không có actor action; c2/c3/c6/c7 kiểm tra đủ actor/pair; action interval không vượt duration; clip quá ngắn bị loại; split theo `split_group` không leakage.
10. Soạn deliverable nghiên cứu cuối cùng thành một markdown duy nhất:
   10.1. Crosswalk từ 2 paper sang hai entrypoint Holistic runtime và batch.
   10.2. Đánh giá manifest theo đúng vai trò batch/export, gồm cột giữ nguyên, cột bổ sung, cột suy ra.
   10.3. Schema `frame_track.csv` và `window_features.csv`.
   10.4. Thiết kế nâng cấp tracking, contract không đổi và rủi ro identity switch.
   10.5. Danh sách feature XGBoost, thứ tự triển khai, và verification cases.
11. Sau khi tracking, tests và toàn bộ plan pass, viết `c:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\context.md` làm tài liệu chốt: runtime flow, foreground policy, state/assignment boundary, CSV/model boundary, invariants, test commands, và ByteTrack decision. Đây là bước cuối, không làm trước implementation và verification.

**Relevant files**
- `c:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\main\main.py` — webcam entrypoint; hiện nhận tracks/landmarks và prediction theo frame, cần chuyển hoặc bổ sung temporal-window inference theo contract mới.
- `c:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\holistic\landmark\landmarks.py` — nguồn sự thật cho nhóm pose/hand/face landmarks MediaPipe Holistic.
- `c:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\holistic\feature_csv\feature_csv.py` — fixed frame-track CSV schema và `MODEL_FEATURE_COLUMNS`; cần giữ compatibility hoặc version rõ khi thêm window contract.
- `c:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\holistic\batch_dataset\batch_dataset.py` — batch entrypoint duy nhất trong pipeline Holistic; đọc manifest và stream CSV train/val/test.
- `c:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\tracking\tracker\tracker.py` — nơi nâng identity association và bbox stability.
- `c:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\tracking\manager\manager.py` — session/assignment/remap/persistence, giữ nguyên ranh giới domain.
- `c:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\pose_gaze\tracking\schemas\schemas.py` — handoff contract cần bảo toàn.
- `c:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze\spec.md` — pair reasoning A→B/B→A, quality gating, temporal state machine.
- `c:\Real-Time-Exam-Proctoring-System\RESEARCH.md` — note nghiên cứu hiện có; chỉ dùng để đối chiếu, không coi Stage 1/2 là runtime dependency.
- `c:\Real-Time-Exam-Proctoring-System\data\processed\stage0_manifest.csv` — manifest đầu vào của batch Holistic, không phải webcam input.
- `c:\Real-Time-Exam-Proctoring-System\R4_CheatingVideoDescriptionBasedonSequencesofGestures_rev2 (2).pdf` — paper 2017 trọng tâm cho gesture chunking và interaction modeling.
- `c:\Real-Time-Exam-Proctoring-System\Exploring_the_potential_of_skeleton_and_machine_le (5).pdf` — paper phụ cho skeleton/landmark feature engineering.

**Verification**
1. Xác nhận mọi claim trong báo cáo cuối cùng truy ngược được về một trong hai PDF, `spec.md`, runtime Holistic code, hoặc `stage0_manifest.csv`.
2. Kiểm tra webcam runtime không import hoặc đọc Stage 1/2 artifact; chỉ batch Holistic mới đọc manifest.
3. Kiểm tra `frame_track.csv` có duy nhất một hàng cho mỗi `frame_id + track_id`, không mất `student_id`, quality flags, landmark validity mask, hoặc timestamp.
4. Kiểm tra `window_features.csv` có frame_count/valid_frame_count, source/peer direction, không leakage giữa split/session, và xử lý đúng thiếu frame.
5. Kiểm tra XGBoost window feature contract khớp với model metadata; nếu model hiện tại chỉ nhận frame features thì phải version/đổi rõ, không âm thầm đổi vector.
6. Kiểm tra tracker giữ identity continuity qua mất detection ngắn, occlusion và giao nhau; đồng thời bbox jitter không làm crop Holistic rung quá mức.
7. Kiểm tra đề xuất manifest mới không xóa hoặc đổi nghĩa cột hiện có.
8. Chỉ tạo `context.md` sau khi tracking tests, overlay/coordinate checks, và verification pass; kiểm tra tài liệu chỉ ghi behavior đã xác minh.
9. Tracking-specific validation must run before any data regeneration: focused tracker tests, detector parsing/NMS tests, duplicate/foreground cases, existing assignment protection, grace-period remap, and coordinate/overlay checks. Full batch/XGBoost validation follows only after these pass.


**Decisions**
- Paper trọng tâm: `R4_CheatingVideoDescriptionBasedonSequencesofGestures_rev2 (2).pdf`.
- Paper phụ: `Exploring_the_potential_of_skeleton_and_machine_le (5).pdf`.
- Chỉ nghiên cứu và thiết kế trong phạm vi `backend/ai_services/pose_gaze`, `data/processed`, và 2 PDF đã chốt.
- `backend/ai_services/pose_gaze/pose_gaze` là pipeline Holistic duy nhất; gồm webcam `main` và batch `holistic/batch_dataset`.
- Stage 1/2 không chạy và không phải dependency; không dùng `windows.parquet`/`selected_frames.parquet` trong thiết kế mới.
- Manifest là input của batch Holistic để tạo frame/window CSV; webcam realtime không đọc manifest.
- Manifest cần additive metadata cho media type, camera view, FPS verification, annotation confidence và directional interaction pairs; frame count/dimensions/FPS thực tế nên đo từ media, không nhập tay.
- 20 video mới có thể đưa vào train/test qua normalized manifest, ưu tiên video gốc; thư mục `data/processed/train|val|test` hiện là Stage 1 artifacts, không phải source chính của pipeline mới.
- Batch cần normalized manifest và validation cho c1-c7, actor rules, action intervals, short clips, class coverage và split leakage; không random frame split.
- XGBoost model chính học trên temporal-window features; frame CSV là dữ liệu trung gian/để debug và tái tạo window.
- CSV contract tách thành `frame_track.csv` và `window_features.csv`.
- `Frame Holistic` là đơn vị xử lý từng frame, phù hợp webcam realtime; `Batch Holistic` là execution mode offline, phù hợp tạo dữ liệu và huấn luyện. Không chọn một để loại cái kia.
- Batch xử lý frame tuần tự rồi tạo window; realtime xử lý frame tuần tự rồi dùng rolling window. Hai entrypoint dùng chung landmark/tracking/schema contract.
- Nâng tracking trong `pose_gaze/pose_gaze/tracking`, tái sử dụng schema/contract hiện có; ưu tiên identity continuity trước bbox stability.
- Với manifest/csv, ưu tiên đề xuất additive và versioned, không phá schema hiện trạng.
- `context.md` được phép tạo, nhưng phải là bước cuối sau implementation, focused tests, và verification; tài liệu không được dùng để che behavior chưa kiểm chứng.

**Domain glossary**
- **Holistic pipeline**: toàn bộ luồng MediaPipe Holistic trong `pose_gaze/pose_gaze`, không bao gồm Stage 1/2.
- **Runtime pipeline**: webcam stream trong `pose_gaze/pose_gaze/main`.
- **Batch Holistic**: `pose_gaze/pose_gaze/holistic/batch_dataset`, đọc manifest và sinh CSV offline.
- **Tracking core**: `tracking/tracker`, `tracking/manager`, `tracking/schemas`; nơi duy trì association và identity contract.
- **track_id**: định danh tạm của tracker trong một session; không phải danh tính học sinh.
- **student_id**: định danh do người dùng/session assignment gán cho track; phải được giữ qua remap.
- **frame-track sample**: một hàng cho một frame và một track.
- **window sample**: một hàng temporal cho một directional relation source→peer.

**Further Considerations**
1. Cần version model metadata khi chuyển từ frame-level `MODEL_FEATURE_COLUMNS` sang window-level XGBoost; không cho runtime nạp nhầm model cũ.
2. Với `c2`, `c3`, `c6`, `c7`, window sample nên giữ source/peer direction; với `c1` và `c4`, taxonomy phải tách tuyệt đối.
3. Tracking có thể dùng motion prediction/appearance or seat-zone constraints sau khi test baseline IoU, nhưng mọi biến thể phải trả cùng `TrackedPerson`/`TrackPacket` contract.


---

## Grilling Review (2026-08-04)

### Confirmed Architecture
- `pose_gaze/pose_gaze/main` is the webcam runtime: detector → `TrackingManager` → Holistic → prediction.
- `pose_gaze/pose_gaze/holistic/batch_dataset` is the offline batch entrypoint in the same Holistic pipeline; it reads manifest and writes CSV.
- `pose_gaze/pose_gaze/tracking` is the shared tracking core: `IoUPersonTracker`, `TrackingManager`, and schemas.
- Stage 1/2 are excluded from the new pipeline and must not be imported as runtime dependencies.

### Evidence-Based Correction
- Stage 2 does not contain a distinct “Stage 2 tracker”; it instantiates the same runtime `IoUPersonTracker` and imports the same schemas/detector wrappers.
- Therefore “reuse Stage 2 tracking” means improve or reuse the shared tracking core, not copy a Stage 2-only algorithm.
- The prior plan's `windows.parquet`/`selected_frames.parquet` and pairwise parquet proposal was removed from the required architecture.

### Grilling Decisions
- Manifest is required by batch Holistic, not by webcam realtime.
- XGBoost target granularity is temporal window; current frame-level feature vector must be versioned or wrapped by a window builder.
- CSV output is two contracts: frame-track rows and window-feature rows.
- Tracking acceptance priority is identity continuity first, bbox/crop stability second.

### Remaining Implementation Risks
- A window-level XGBoost model cannot silently replace the current frame-level model; model metadata and inference path need an explicit version boundary.
- `IoUPersonTracker` is greedy IoU and has no motion/appearance prediction; occlusion and crossing tests are required before claiming identity continuity.
- Manifest video-level intervals need a batch-local frame/window expansion step inside `holistic/batch_dataset`; this must not depend on Stage 1.


## Tracking Incident Review (2026-08-04)

### Observed failures
- Failure A: two bounding boxes can be emitted for one person. Current `UltralyticsPersonDetector.detect()` forwards model boxes with default prediction settings; current tracker accepts both as separate candidates after one matches an existing track.
- Failure B: landmarks do not follow real movement. Most likely first-order cause is wrong/duplicated track or crop identity, not proven letterbox math failure. `HolisticLandmarkExtractor` owns one MediaPipe processor per `track_id`, so duplicate tracks split one person's temporal smoothing into separate processors.
- Current `_LetterboxTransform` maps fixed square task canvas back to changing crop using `content_width`, `content_height`, and padding. This inverse is internally consistent; do not replace it before a coordinate regression test proves transform error.

### Proposed fix order
1. Instrument raw detections, deduplicated detections, tracks, crop boxes, and landmark overlays in separate colors/log fields. Confirm whether screenshot boxes are raw detections or tracked boxes.
2. Detector boundary: pass explicit Ultralytics NMS `iou` and `max_det`; tune on close-seated two-person clips. Add post-parse duplicate suppression using high overlap plus containment/center proximity, not IoU alone, so two real students are not merged.
3. Tracker boundary: keep one canonical track when duplicate detections survive. Preserve canonical `track_id` and `student_id`; close/remove duplicate state and its Holistic processor. Do not let duplicate detection create a second slot.
4. Association boundary: add predicted center/bbox velocity and gate matches using IoU plus normalized center distance, area/size ratio, and optional fixed seat-zone constraints. Use seat zones after manual student assignment to prevent cross-person identity switches. Keep `TrackedPerson` and `TrackPacket` contract unchanged.
5. Crop boundary: validate crop includes head, shoulders, and hands. Stabilize crop only after association passes; optional EMA may be applied to crop bbox for input stability while raw detection bbox remains recorded for audit.
6. Landmark boundary: retain current letterbox transform until synthetic point round-trip and real-frame overlay tests pass/fail. Test `frame_x/frame_y` against original frame coordinates, processor key against stable student identity, timestamp monotonicity, and landmark jump after controlled bbox jitter. Tune soft confidence only after separating missing points from coordinate displacement.

### Acceptance tests
- Duplicate detection test: one person with two overlapping detections yields one present track and one Holistic processor key.
- Two-person test: two seated students remain two tracks when boxes overlap moderately; no suppression based on IoU alone.
- Occlusion test: short disappearance restores same student identity without creating a duplicate track.
- Crossing/near-contact test: seat-zone or motion gate prevents student ID swap.
- Overlay test: displayed landmarks lie on the same person as `crop_bbox`; raw bbox, crop bbox, and landmark colors are distinguishable.
- Coordinate test: synthetic landmark through letterbox and inverse mapping returns expected frame pixel within tolerance.

### Boundary decision
- Stage 2 has no separate superior tracker. It reuses shared runtime `IoUPersonTracker`; improvements belong in `pose_gaze/pose_gaze/tracking` and shared detector adapter.
- Regenerate any offline landmark/training artifacts only after runtime tracking and overlay tests pass. Do not use old Stage 2 artifacts as evidence that new runtime tracking works.


## Foreground Selection Decision (2026-08-04)

### Evidence
- YOLO-related paper `data-07-00122 (2).pdf`, pp. 2-3, discusses viewpoint, occlusion, scale, illumination, clutter, camera motion, and brightness as factors affecting action recognition quality. It does not define a special method for selecting the two foreground students.
- Stage 2 foreground behavior comes from project code: `dataset/stage2/common.py::MAX_TRACKS = 2` and `dataset/stage2/export_csv.py::_select_front_tracks()`, which requires confidence >= 0.5, bbox area ratio >= 0.01, sorts by bbox area, and keeps up to two tracks.
- Runtime regression `test_high_confidence_outsider_does_not_hide_existing_students()` protects existing assigned tracks by associating all detections before filling new slots. This behavior must remain.

### Decision
- Preserve foreground selection as a first-class runtime policy. Do not replace it with ByteTrack alone.
- Runtime pipeline: detector class/confidence gate, duplicate suppression, associate existing tracks first, rank only new-track candidates by foreground score, cap at two active student tracks, then apply student assignment/seat-zone constraints, then Holistic.
- Foreground score should reuse Stage 2 signals (bbox area ratio and confidence) and add fixed seat/ROI zone and bottom-center plausibility when available. Area alone is not sufficient.
- ByteTrack may become an association backend behind the same `TrackedPerson`/`TrackPacket` contract, but must receive the foreground cap/seat-zone policy and must not create arbitrary background tracks.
- Do not claim YOLO paper as source for two-student foreground selection; cite it only for image-quality risks and detector robustness concerns.

### Acceptance tests
- Three detections: two foreground students plus background person. Existing students remain; background never occupies a student slot when foreground score is lower.
- One student with duplicate boxes: one canonical track only.
- High-confidence background outsider: does not displace existing students.
- Initial frame with background confidence higher than student: foreground/seat-zone policy still selects intended students.
- Two real students with moderate bbox overlap: both survive duplicate suppression.

## Tracking Update Specification

### Runtime invariants
- At most two active student tracks per session/frame.
- Existing assigned/present tracks receive association priority; new detections cannot displace them because of confidence sorting alone.
- One physical person cannot produce two canonical tracks from overlapping/contained detections.
- Two real students with moderate overlap must remain distinct; duplicate suppression must use overlap plus containment/center/size evidence, never IoU alone.
- `track_id`, `student_id`, `TrackedPerson`, and `TrackPacket` contracts remain stable. Invalid normalized coordinates use masks and finite fallback values; no NaN/Infinity.
- Holistic processor state remains keyed to canonical `track_id`; duplicate merge/removal must not create a second processor for same person.

### Implementation order
1. Add detector telemetry and explicit Ultralytics `iou`/`max_det` controls without changing class contract.
2. Add deterministic duplicate suppression before track creation; cover duplicate, nearby-student, and contained-box cases.
3. Update `IoUPersonTracker` association with existing-track priority and bounded center/area/motion gates; preserve grace-period remap behavior.
4. Add runtime foreground ranking/cap using Stage 2 signals (`confidence`, bbox area ratio) plus optional configured seat/ROI plausibility. Keep `_select_front_tracks()` as historical/export reference, not runtime dependency.
5. Add crop/landmark overlay and coordinate round-trip checks only after identity behavior passes.
6. Benchmark optional ByteTrack adapter against baseline IoU. Do not add dependency or enable backend by default until max-two and foreground tests pass.
7. Rebuild offline frame/window artifacts only after runtime tracking passes; retrain/version XGBoost separately.
8. Write `backend/ai_services/pose_gaze/context.md` last, recording only verified behavior, thresholds, contracts, test commands, and unresolved risks.

### Focused test matrix
- Detector NMS/max detection settings are passed and parsed correctly.
- Duplicate boxes for one person yield one canonical present track and one Holistic processor key.
- High-confidence outsider does not displace existing assigned students.
- Initial frame ranks intended foreground students despite a larger/high-confidence background candidate when ROI/seat policy identifies them.
- Two real students with moderate bbox overlap survive deduplication and remain assigned separately.
- Short disappearance restores same track/student during grace period.
- Crossing/near-contact does not swap student assignment when seat-zone or motion gate is configured.
- Raw bbox, crop bbox, and landmark overlays identify same person; synthetic letterbox inverse returns frame coordinates within tolerance.

- ByteTrack comparison: same foreground-selection test suite passes before measuring identity continuity/occlusion gains.

## Bbox Smoothing and Crop Stabilization Addendum

### Findings
- `IoUPersonTracker.update()` currently replaces matched track bbox with raw detector bbox. Downstream crop and landmark mapping inherit detector jitter.
- Association must continue using current track state and raw detection geometry. Smoothing belongs after accepted match.
- Crop padding, letterboxing, landmark schema, and `TrackedPerson`/`TrackPacket` contracts remain unchanged.

### Implementation
1. Add validated `bbox_smoothing_alpha` to `IoUPersonTracker`; use `1.0` as no-smoothing baseline and test `0.5` as initial candidate.
2. On accepted match, smooth each bbox edge with EMA and publish smoothed bbox. Keep raw detection local for association and telemetry.
3. Initialize new tracks from raw detection. Retain last smoothed bbox during missed frames. Discard it after expiry. Restore persisted bbox unchanged.
4. Propagate setting through `PersonTrackingConfig`, `TrackingManager`, video/webcam entrypoints, and batch tracker construction.
5. Apply smoothing only to continuous video/clip sequences. Never reuse tracker state across unrelated independent images.

### Acceptance Tests
- One-frame detector jump moves published bbox partially, then converges.
- Missed frame retains last smoothed bbox; reappearance continues from it.
- Expired track creates new ID with raw detection bbox.
- Alpha validation rejects invalid values.
- Crossing, outsider, duplicate, persistence, and foreground tests remain passing.
- Crop remains non-empty with positive dimensions; frame coordinates remain valid.

### EMA versus Kalman
- Use EMA first. Current problem is detector jitter in fixed-camera, seated-student footage; Kalman adds velocity, covariance, process-noise, measurement-noise, and miss-prediction tuning.
- Defer Kalman until benchmark shows EMA lag during real movement or long occlusion. Compare bbox edge error, ID continuity, pose-valid ratio, empty crops, missed frames, and FPS.
- Kalman must not hide association or crop-semantic errors.

### Verification Benchmark
1. Run identical 60-frame video with alpha `1.0` and `0.5`.
2. Compare per-track adjacent bbox/crop edge deltas, empty crops, pose-valid frames, missed frames, ID swaps, and FPS.
3. Accept smoothing only when jitter decreases without pose loss, identity regression, empty crops, or unacceptable FPS regression.
4. Do not regenerate full dataset until focused tracker, crop, landmark, overlay, and benchmark checks pass.

## World Landmark Ablation Addendum

- Do not delete isolated `pose_world[16]` or `right_hand_world[0]`; both are wrist anchors within complete landmark groups. Equal `x/y` values do not prove interchangeable coordinate frames.
- Current serialization keeps world `x/y` but drops `z`; evaluate whether world features retain enough 3D value.
- Compare two feature sets after bbox/crop stabilization:
   1. Full current schema: 2D pose/hands/face plus `pose_world`, `left_hand_world`, and `right_hand_world`.
   2. 2D-only schema: remove all world landmark groups together.
- Compare validation/test F1 or AUC, per-class recall, missing-feature ratio, feature count, inference cost, and model stability.
- Removing world groups requires updating `LANDMARK_GROUPS`, `MODEL_FEATURE_COLUMNS`, CSV headers/schema version, model metadata, and retraining/re-exporting the model. Do not remove runtime output while retaining old model feature names.
- If world features remain, evaluate preserving `z`; current `LandmarkPoint` and CSV `POINT_FIELDS` retain only `x/y/visibility/presence`.
- Before benchmark, keep existing schema unchanged. No isolated landmark deletion.
