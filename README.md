# Hệ thống giám sát phòng thi bằng AI (Exam Proctoring System)

> Tài liệu này mô tả **kiến trúc thực tế đang chạy trong code**, không mô tả thiết kế ước nguyện.
> Mọi nhận định dưới đây đều kèm đường dẫn `file:dòng` và đã được kiểm chứng bằng cách chạy thử
> bản Ubuntu trên máy Intel Core Ultra 7 155H (iGPU Arc) ngày 2026-10-27 — xem [§8](#8-đã-kiểm-chứng-bằng-chạy-thật).

---

## 1. Tóm tắt

Hệ thống giám sát hành vi gian lận **trong một tiến trình Streamlit duy nhất**. Video vào/ra qua
**WebRTC** (`streamlit_webrtc`), 4 module AI chạy ở **4 daemon thread riêng**, trao đổi với nhau bằng
`queue.Queue`. Micro **không** đi qua WebRTC mà được PyAudio/`sounddevice` chiếm trực tiếp.

Bốn module đang bật:

| Module | Model thật | Phát hiện |
| :-- | :-- | :-- |
| Vật cấm trên bàn | YOLO fine-tuned (`weights/best (1).pt` / bản OpenVINO) | `earphone`, `smartphone`, `smartwatch` |
| Người lạ thay thi | insightface: RetinaFace `det_10g` + ArcFace `w600k_r50` (512-d) | khuôn mặt không khớp DB |
| Giấy thi / phao | YOLO person (`yolov8n`) + YOLO giấy + `CountBasedPaperMonitor` | tờ giấy mới vượt baseline của từng thí sinh |
| Hành vi & giọng nói | MediaPipe Holistic + 3 XGBoost causal (c2/c3/suspicious) + PhoWhisper + PhoBERT | trao đổi, quay ngó, nói gian lận |

**Không còn FastAPI backend.** `backend/main.py:6-14` (khởi tạo app, `include_router`, `/health`) đã bị
comment toàn bộ; `backend/api/` chỉ còn `pose_gaze_routes.py` không được ai mount. Thư mục `frontend/`
như tài liệu cũ mô tả **không tồn tại** — app nằm ngay ở thư mục gốc.

Từ bản này, phần logic phát hiện trên từng khung hình đã tách sang `backend/ai_services/pipeline/` để
**dùng chung** giữa 2 app realtime và `run_offline.py` (video vào → video burn cảnh báo + báo cáo ra,
xem [§11](#11-luồng-offline-run_offlinepy)).

---

## 2. Kiến trúc thực tế

### 2.1 Đường đi của một khung hình

```
Browser (getUserMedia)
   │  WebRTC SENDRECV, audio:False
   ▼
video_frame_callback()                 app_webrtc.py:422   (chạy trên event loop, PHẢI nhanh)
   │  1. av.VideoFrame -> numpy BGR
   │  2. xúc RESULT_QUEUE -> ACTIVE_OVERLAYS + SHARED_STATE["logs"]
   │  3. bơm frame copy vào 3 queue theo stride
   │  4. draw_warning_overlays() -> trả frame đã vẽ về browser
   ▼
4 thread AI  ──(kết quả dict)──►  RESULT_QUEUE (maxsize=100)
```

Frame luôn **bỏ frame cũ nhất** khi queue đầy (`maxsize=2` + `get_nowait()` trước khi `put`) — ưu tiên
độ trễ thấp hơn là không rớt frame.

### 2.2 Phân phối frame (`app_webrtc.py:459-474`)

| Queue | Stride | Việc làm |
| :-- | :-- | :-- |
| `GAZE_QUEUE` | **mọi frame** (`% 1`) | pose/gaze + tracking + holistic + causal classifier |
| `PAPER_QUEUE` | mỗi 5 frame | đếm giấy theo người |
| `FRAME_QUEUE` | mỗi `FPS_SKIP = 1` frame → **mọi frame** | YOLO vật cấm + ArcFace verify |

`FPS_SKIP = 1` (`app_webrtc.py:406`) đang ghi đè mất `settings.object_detect_every_n_frames = 2`
(`backend/core/config.py:44`) ở tầng ứng dụng — và cũng khác luôn `FPS_SKIP = 5` khai báo ở
`config.py:7` (hằng số trùng tên nhưng app không import). Đây là điểm nóng nhất về hiệu năng của hệ thống.

### 2.3 Bốn luồng AI (`init_system_resources`, `app_webrtc.py:118`)

Tất cả chạy trong `@st.cache_resource` → **chỉ khởi tạo một lần cho cả tiến trình**. Bốn `Event`
(`vision_ready`, `paper_ready`, `audio_ready`, `gaze_ready`) được `wait()` trước khi hàm trả về
(`app_webrtc.py:396-399`), nghĩa là **trang web bị block ở trạng thái "Running…" cho tới khi cả 4 model
nạp xong** (vài phút ở lần chạy đầu, xem §8).

| Luồng | Dòng | Nạp gì | Đẩy vào `RESULT_QUEUE` |
| :-- | :-- | :-- | :-- |
| `heavy_vision_thread` | `:125` | `ObjectDetector` + `FaceVerifier` | `module="object_detect"`, `module="face_verify"` |
| `paper_vision_thread` | `:206` | `UltralyticsPersonDetector` + `ObjectDetectModule` + `PaperCountPipeline` | `module="paper_count"` |
| `fast_gaze_thread` | `:285` | `PersonTrackingModule` + `HolisticLandmarkExtractor` + causal classifier (`gaze_scan.build_stack`) + `PoseGazeDetector` | `module="pose_gaze"` |
| `audio_ai_thread` | `:319` | `RealtimeAudioWorker` (PyAudio/sounddevice + VAD + PhoWhisper + PhoBERT) | `module="audio_phobert_pipeline"` |

### 2.4 Package dùng chung cho mọi runtime: `backend/ai_services/pipeline/`

Phần logic "1 frame vào → alert ra" không còn nằm trong hai file app nữa. Mỗi module là hàm thuần,
**không thread, không queue, không `st.`**, nên realtime và offline gọi cùng một code:

| Module | Hàm | Realtime gọi ở | Offline gọi ở |
| :-- | :-- | :-- | :-- |
| `events.py` | `to_log_line` `:16`, `prune_overlays` `:78`, `expire_overlays` `:87` | `app_webrtc.py:438` / `:446` / `:415` | `run_offline.py:149` / `:166` / `:172` |
| `overlays.py` | `draw` `:26` (+ `log_lines` để burn cột lịch sử) | `app_webrtc.py:417` | `run_offline.py:173` |
| `face_scan.py` | `scan_stranger_with_masking` `:18`, `crop_for_registration` `:57` | `app_webrtc.py:169` / `:185` | `run_offline.py:278` |
| `object_scan.py` | `scan_objects` `:14` (lọc nhãn giấy khỏi vật cấm) | `app_webrtc.py:217` | `run_offline.py:276` |
| `paper_scan.py` | `arm_baseline` `:12`, `build_alert` `:23` | `app_webrtc.py:275` / `:303` | `run_offline.py:340` / `:350` |
| `gaze_scan.py` | `build_stack` `:79`, `scan_frame` `:134` (trả **list** alert, mỗi actor một alert) | `app_webrtc.py:318` / `:339` | `run_offline.py:262` / `:288` |
| `audio_scan.py` | `DynamicSpeechChunker` `:28`, `is_alert` `:121`, `to_alert` `:127` | `realtime_audio_ubuntu.py:49` + `:98`; `realtime_audio_worker.py:28` | `run_offline.py:97` (chunker) + `:120` (`to_alert`) |

Hai hàm cũ trong app giữ nguyên chữ ký nhưng chỉ delegate — `get_largest_stranger_with_masking`
(`app_webrtc.py:168-169`) và `draw_warning_overlays` (`app_webrtc.py:411-417`) — nên không phải sửa các chỗ gọi.
Riêng `face_scan.crop_for_registration` và `audio_scan.is_alert` **không** có chỗ gọi phía offline (không có
nút đăng ký mặt; audio file luôn đi qua `to_alert`).

> Docstring trong `pipeline/*.py` kiểu *"Sao y app_webrtc.py:255-302"* tham **số dòng của bản app trước
> khi tách module** — dùng để truy nguồn gốc logic, không phải vị trí hiện tại.

---

## 3. Chi tiết từng module

### 3.1 Vật cấm — `ObjectDetector` (`backend/ai_services/object_detect/object_detect.py`)

- `BANNED_ITEMS = ["earphone", "smartwatch", "smartphone"]` (`object_detect.py:27`) — `cheat_sheet` đang bị
  comment, nên giấy **không** đi qua module này.
- Có bảng id→tên cứng cho trường hợp OpenVINO mất `.names`: `{0: smartphone, 1: cheat_sheet, 2: earphone, 3: smartwatch}` (`object_detect.py:159`).
- Trả về `None` hoặc đúng `{module, status, timestamp, details, detections}`; mỗi detection là
  `{label, confidence, bbox}` với `bbox = [x1,y1,x2,y2]` trên **frame gốc, không resize**. `status` luôn `"alert"`.
- Ứng viên `smartphone` bị loại nếu `_looks_like_calculator` / `_looks_like_pen` (`object_detect.py:189-202`).
- Ở tầng app, mọi nhãn giấy (`paper/document/cheatsheet/sheet/tai_lieu/giay/book`) bị lọc bỏ lần nữa để
  không đúp báo với pipeline giấy (`pipeline/object_scan.py:11-28`, `PAPER_LABELS`).

### 3.2 Người lạ — `FaceVerifier` (`backend/ai_services/face_verify/face_verify.py`)

- **Không phải FaceNet/face_recognition** như tài liệu cũ. Dùng **insightface** qua ONNX Runtime, nhưng
  **không** nạp `FaceAnalysis` (5 model) mà tự chọn đúng 2 file: `det_10g.onnx` (RetinaFace) và
  `w600k_r50.onnx` (ArcFace, 512-d) từ pack `buffalo_l` trong `~/.insightface/models` (`face_verify.py:78-110`).
- `db_path="data/student_faces/"`: **`person_id` = tên file không phần mở rộng** (`SV001.jpg` → `SV001`).
  `_load_database()` giữ khuôn mặt lớn nhất mỗi ảnh, cache `normed_embedding`, tự bật FAISS khi DB lớn.
- So khớp: **cosine similarity** (`_known_vectors @ query`), ngưỡng `settings.face_similarity_threshold = 0.4`
  (`backend/core/config.py:143`). Dưới ngưỡng → `status="alert"`, `message="unauthorized_person"`,
  `details={similarity_score, unauthorized_bbox}` (`face_verify.py:187-213`).
- Mẹo **masking** (`pipeline/face_scan.py:18-55`): vẽ ô đen lên mọi khuôn mặt khác rồi verify từng mặt,
  để `verify_face` (vốn chỉ báo mặt đầu tiên khớp kém) không "ăn gian" kết quả giữa nhiều người.
  Trả `unauthorized_bbox` gốc về cho UI vẽ khung vàng "NGUOI LA".
  ⚠️ Giá phải trả: mỗi frame gọi `_detect_faces` `1 + N` lần (`verify_face` tự detect lại bên trong).
  Cùng module này có `crop_for_registration()` (`:57`) dùng cho nút đăng ký mặt (+40px quanh khuôn lớn nhất).

### 3.3 Giấy thi / phao — `PaperCountPipeline` + `CountBasedPaperMonitor`

- `ObjectDetectModule` (`object_detect.py:353+`) chạy YOLO **mỗi 2 frame** và phát lại kết quả cũ với
  `inference_ran=False` (`object_detect.py:586-611`); xác nhận dị thường theo **3/5 frame trùng vị trí**
  (`object_detect.py:1419-1455`). Kết quả nhét vào khoá `object_result` của pipeline.
- `PaperCountPipeline.process_frame()` trả: `session_id, frame_id, timestamp_ms, people, papers,
  paper_count_state, paper_monitoring_armed, object_result, alerts, risk_score` (`paper_count_pipeline.py:201-215`).
- Máy trạng thái (`paper_count_monitor.py`):
  - **SETUP** (`armed=False`): gom tối đa 60 snapshot số giấy, baseline = **mode** của chuỗi đó (`:507`).
  - **ARM** (`arm()`, `:80`): đóng băng `baseline_count` + `baseline_clusters`. Nút "🔒 Cố định Baseline (A)"
    gọi `arm_paper_monitoring()`; "🔄 Reset Setup (D)" tạo `session_id` mới (`app_webrtc.py:537-541`,
    logic gộp ở `pipeline/paper_scan.py:12-20`).
  - **Gom cụm**: union-find nhập 2 box khi IoP(phần tử nhỏ) ≥ `0.35` **hoặc** khoảng cách tâm chuẩn hoá
    ≤ `0.70` (`paper_count_monitor.py:285-330`; giá trị lấy từ `config.py:123-124` qua
    `paper_count_pipeline.py:61-66`, **không** dùng default `0.90` của constructor) — tránh gộp nhầm 2 tờ sát nhau.
  - **Baseline theo từng người**: giấy gán cho người gần nhất (`_nearest_person`), rồi so
    `c_count > baseline_owner_counts[owner]` **và** so tổng toàn cục; cần
    `paper_count_confirm_inferences = 2` lần inference liên tiếp mới báo (`:152-218`).
    Đây là bản vá lỗi "đếm toàn cục" mà tài liệu cũ mô tả.
- Mỗi phần tử `papers`: `{observation_index, bbox_xyxy, confidence, class_names, source_box_count,
  status, owner_track_id, owner_person_id}`; `status ∈ {baseline_paper, suspicious_new_paper}`.
  UI vẽ đỏ `CHEAT SHEET (NEW)` / vàng `Paper #n` (`pipeline/overlays.py:70-78`).
- Cảnh báo: `{source:"paper_count", label:"cheat_sheet", reason:"paper_count_above_baseline",
  risk_score:1.0, bbox_xyxy, owner_track_id, owner_person_id}`; `paper_scan.build_alert()` **vứt** alert
  `smartphone_detected` phát sinh từ module này (`paper_scan.py:36-40`).
- Bật tắt: `smartphone_fallback_enabled = False` (`config.py:45`) và cả hai runtime còn force-tắt trong
  pipeline giấy (`app_webrtc.py:249-256`, `run_offline.py:251-257`) ⇒ nhánh COCO fallback
  (`object_detect.py:839`) là **code chết**.

### 3.4 Hành vi — tracking + Holistic + causal classifier

Toàn bộ stack này nằm trong `pipeline/gaze_scan.py`: `build_stack()` (`:79-131`) nạp
`PersonTrackingModule` + `HolisticLandmarkExtractor` + classifier, `scan_frame()` (`:134-176`) chạy một
frame và **trả về danh sách alert, mỗi actor một alert** — nên realtime (`app_webrtc.py:318/339`) và
offline (`run_offline.py:262/288`) dùng chung.

`PoseGazeDetector` (`pose_gaze_service.py:83-141`) **không tự tính góc Euler**; nó là bọc ngoài của
cùng một stack: YOLO person tracking → MediaPipe Holistic theo crop từng người → XGBoost causal.
Trả `None` hoặc `{module:"pose_gaze", status:"warning", timestamp, details:{action, actor_id, score}}`.

- **Tracking** (`backend/ai_services/pose_gaze/tracking/`): `track_id` (số, tạm) vs `student_id`
  (định danh gán bởi giám thị, `manager.assign_student()`). `packet = TrackPacket(session_id, frame_id,
  timestamp_ms, tracks)`; track có `bbox, confidence, age_frames, missed_frames, is_present, student_id`.
  App tự đặt `student_id = "student_{track_id:02d}"` (`gaze_scan.py:141-147`).
  Session lưu `test_data_tracking/<session_id>/tracking_state.json` mỗi 30 frame
  (`state_persist_interval_frames=30`, `tracking/manager.py:68`), nhưng app tạo `session_id` mới theo timestamp
  mỗi lần chạy ⇒ **state được ghi ra nhưng không bao giờ được restore**.
- **Holistic** (`.../pose_gaze/holistic/landmark/landmarks.py`): một MediaPipe Holistic context **riêng cho
  mỗi track_id**, chạy trên crop đã nới `crop_padding=0.15`; `soft_landmark_confidence=0.2` giữ toạ độ x/y
  nhưng xoá visibility của điểm yếu; `face_hold_frames=3` giữ khuôn mặt 3 frame sau khi mất.
  Landmark được trích **ngay trong frame đang xử lý** (đúng yêu cầu real-time, không dùng JSON tiền xử lý).
- **Causal classifier** (`.../test_media/live_actor.py:429`): **không phải neural sequence model**.
  Đó là cửa sổ trượt nhân-quả theo từng actor + tổng hợp quantile + **3 Booster XGBoost**
  (`models/c2.ubj`, `models/c3.ubj`, `models/suspicious_activity.ubj`) trong bundle
  `stage6_bundle_exact/causal_8fps_stage6_mixed_084699_final_20260827/`, kèm `calibration.json`
  (ngưỡng + gate), `feature_schemas.json` (c2=65, c3=90, suspicious=50 đặc trưng),
  `temporal_policy.json` (`target_fps=8`, `baseline_valid_frames=4`, `window_frames=24`).
  `update()` trả `{actor_id: {predicted_class, candidate_class, current_scores, current_gates,
  c2_score, c3_score, suspicious_activity_score, warmup_frames_seen/required, evidence_*, causal:True, ...}}`.
- **Lớp tiếp cận được: `c2`, `c3`, `suspicious_activity`, `c5`** — ưu tiên
  `{"c5":0, "suspicious_activity":1, "c3":2, "c2":3}` (`causal_stream.py:319`); điểm cao hơn mới ghi đè,
  frame âm sau không kéo về `c5`. **Việc tách `suspicious_activity` → c1/c4 bằng YOLO chưa được nối vào
  đâu cả** (không có trong `app_webrtc*.py`, `backend/`, `scripts/`, `notebooks/`); `c1_model_dir`,
  `c4_model_dir`, `c7_model_dir` trong `pose_gaze_service.py:61-63` không được đọc.
  Trong `pose_gaze_service.py:116-121` vẫn còn nhãn `c1`/`c4` — **không bao giờ reachable**; điều kiện
  `"warming" not in str(classification)` (`:114`) cũng không lọc được gì vì không khoá nào chứa chuỗi đó
  (warmup thật do `predicted_class == "c5"` chặn).

### 3.5 Âm thanh — VAD + PhoWhisper + PhoBERT

- Cắt câu: `pipeline/audio_scan.py:28-119` (`DynamicSpeechChunker`) — MIN 2.0s, MAX 6.0s, đuôi im lặng
  0.8s, ngưỡng RMS 0.02, overlap 1.5s khi cắt cưỡng bức, `resample_poly` về 16kHz. **Ba** nơi cùng dùng:
  `realtime_audio_ubuntu.py` (PyAudio, paFloat32, chunk 0.5s, bỏ 1.0s đầu làm noise profile),
  `realtime_audio_worker.py` (Windows, sounddevice, né thiết bị "stereo mix/loopback", noise profile = 0)
  và `run_offline.py:70-126` (`decode_audio()` + `transcribe_file_audio()`: giải mã track audio của
    file bằng `av`, noise profile = 0 mặc định).
  Chunker cũng nhớ `last_span` (vị trí tuyệt đối của câu trong nguồn) nên offline gắn đúng mốc video
  cho cảnh báo âm thanh.
- `AudioPipeline.process_audio()` (`audio_pipeline.py:39`): bandpass → **Silero-VAD** (ngưỡng 0.65) →
  **PhoWhisper** (`vinai/PhoWhisper-small`; bản OpenVINO IR ở `backend/ai_services/whisper/weights/phowhisper_ov`,
  `whisper/config.py:2`) → KeywordDetector + **PhoBERT fine-tuned 2 nhãn**
  (`whisper/phobert/weights/model.safetensors`, cache OV tại `weights_openvino/`).
- Trả về: `{module, status: idle|success|alert, transcription, speech_segments, risk: Normal|Suspicious|Cheating,
  confidence, rule_label, ai_label, fusion_reason, matched_keywords, timestamp}` (`audio_pipeline.py:110-142`).
  App đổi `module` thành `audio_phobert_pipeline` và chỉ hiển thị khi `transcription` khác rỗng.
- ⚠️ `audio_pipeline.py:79` kiểm tra `rule_label in ["Suspicious","Cheating"]`, nhưng
  `KeywordDetector.get_risk()` trả `high|medium|low|safe` (`keyword_detector.py:243`) rồi `.capitalize()`
  ⇒ **nhánh keyword không bao giờ tự báo động**; mọi alert hiện tại đến từ PhoBERT.
  `DecisionFusionService` được import nhưng không dùng (`audio_pipeline.py:15`).

---

## 4. Cấu trúc thư mục thực tế

```
Real-Time-Exam-Proctoring-System/
├── app_webrtc.py            # ENTRY POINT DUY NHẤT (Streamlit + WebRTC) — tự nhận Ubuntu/Windows, §5
├── app_webrtc_win.py        # SHIM 12 dòng: `streamlit run app_webrtc_win.py` -> exec file trên
├── run_offline.py           # ENTRY POINT offline: 1 file video -> video burn cảnh báo + báo cáo (§11)
├── offline_job.py           # Bọc run_offline trong luồng nền cho UI (§11.4) — không import streamlit
├── app_ubuntu.py            # LEGACY: cv2.VideoCapture + cửa sổ OpenCV, không WebRTC
├── app_win.py               # LEGACY: như trên cho Windows
├── backend/
│   ├── core/
│   │   ├── config.py        # class Settings (plain class) — ngưỡng, stride, class names
│   │   └── device.py        # resolve_device / resolve_model_path / make_ort_session
│   ├── main.py              # DEAD: FastAPI app đã bị comment (dòng 6-14)
│   ├── api/pose_gaze_routes.py   # DEAD: không được mount
│   └── ai_services/
│       ├── pipeline/        # LOGIC DÙNG CHUNG realtime + offline (§2.4), không thread/không st.*
│       │   ├── events.py    #   alert -> dòng log, giữ/hạn overlay theo TTL
│       │   ├── overlays.py  #   vẽ khung + nhãn + (offline) burn bảng cảnh báo
│       │   ├── face_scan.py #   masking verify người lạ + crop đăng ký mặt
│       │   ├── object_scan.py # YOLO vật cấm, lọc nhãn giấy
│       │   ├── paper_scan.py#   ARM baseline + đóng gói alert giấy
│       │   ├── gaze_scan.py #   tracking + holistic + causal classifier
│       │   └── audio_scan.py#   DynamicSpeechChunker (RMS VAD) + đóng gói alert âm thanh
│       ├── object_detect/   # ObjectDetector, ObjectDetectModule, PaperCountPipeline,
│       │                    #   CountBasedPaperMonitor, các test_*.py
│       ├── pose_gaze/       # pose_gaze_service.py + tracking/ (bản app dùng)
│       │   └── pose_gaze/   #   package nested: holistic/, tracking/ (bản dev, trùng lặp)
│       ├── face_verify/     # FaceVerifier (insightface RetinaFace + ArcFace)
│       └── whisper/         # RealtimeAudioWorker (ubuntu/win), AudioPipeline,
│                            #   PhoWhisper, PhoBERT, keyword/negative rules
├── stage6_bundle_exact/causal_8fps_stage6_mixed_084699_final_20260827/
│                            # bundle causal XGBoost: models/*.ubj + calibration + schemas
├── weights/                 # best (1).pt, best_openvino_model/, yolov8n.pt,
│                            #   yolov8n_openvino_model/, mediapipe/holistic_landmarker.task
├── data/
│   ├── student_faces/       # DB khuôn mặt: 1 ảnh / thí sinh, tên file = person_id
│   ├── registered_logs/     # ảnh toàn cảnh khi đăng ký
│   ├── sessions/            # snapshots object_detect_*.jpg + jsonl
│   └── test_audio/          # clip .m4a/.mp3 test ASR
├── offline_out/<run>/       # SẢN PHẨM offline: annotated.mp4, events.jsonl, log.txt, report.json, tracking/
├── test_data_tracking/      # tracking_state.json của các session
├── scripts/                 # check_env.py (chẩn đoán CUDA/OpenVINO), export_models.py (.pt -> OV/ONNX)
├── docs/                    # adr/ + các technical note về retrain causal 8fps
├── chay.txt                 # ~19 lệnh test thực địa (PowerShell, máy Windows) — đọc lưu ý §10
└── requirements.txt
```

---

## 5. Thiết bị và model

`backend/core/device.py` là tầng chọn thiết bị duy nhất, ưu tiên **CUDA → Intel iGPU (OpenVINO) → CPU**,
ép bằng biến môi trường `PROCTOR_DEVICE = auto | cuda | intel | cpu` (`device.py:8`).
`resolve_model_path()` tự đổi `weights/best (1).pt` → `weights/best_openvino_model/` hoặc `best.onnx`
tuỳ thiết bị (`device.py:141`).

**Chỉ còn một entry point.** `app_webrtc_win.py` chỉ còn 12 dòng `runpy.run_path()` để lệnh
`streamlit run app_webrtc_win.py` không vỡ, còn cấu hình thiết bị nằm trong một bảng tra theo
**kind của thiết bị đã resolve** (`app_webrtc.py:83-93`), không phải theo OS:

| `select_device().kind` | Model vật cấm | `device` ObjectDetector | `PERSON_DEVICE` (paper + tracking) |
| :-- | :-- | :-- | :-- |
| `cuda` (NVIDIA, kể cả WSL) | `weights/best (1).pt` | `None` | `"auto"` |
| `intel` (iGPU Arc/HD) | `weights/best_openvino_model` | `"GPU"` | `"cpu"` |
| `cpu` | `weights/best (1).pt` | `None` | `"cpu"` |

Ba dòng trên chính là giá trị mà hai file entry point cũ vẫn dùng (`app_webrtc.py:84-86`), nên
Windows+NVIDIA và Ubuntu+iGPU **không đổi hành vi**; khác biệt là WSL+NVIDIA giờ lấy profile `cuda`
thay vì bị kéo vào nhánh Ubuntu (trước đó paper + tracking bị ép chạy CPU dù có RTX).
`VISION_MODEL_PATH`/`PERSON_DEVICE` được dùng ở `:157-158` và `:237`, `:239`, `:318`.
Ép ngược bằng `PROCTOR_DEVICE=cuda|intel|cpu` vì `select_device()` đọc biến đó (`device.py:130`).

**Chỉ còn một thứ phân nhánh theo OS: worker micro** (`:105-108`) — Windows dùng
`realtime_audio_worker.py` (sounddevice), Linux/WSL dùng `realtime_audio_ubuntu.py` (PyAudio), vì
cả hai đều là chuyện thiết bị âm thanh của hệ điều hành, không liên quan GPU.

> Ghi chú: `device="GPU"` vẫn hoạt động vì `ObjectDetector` tự gọi `resolve_device()` — alias `"gpu"`
> được chuẩn hoá về `auto` (`device.py:96`). `gaze_scan.build_stack()` cũng tự chuẩn hoá thiết bị trước
> khi đưa cho ultralytics (`gaze_scan.py:89-91`) — đây là lỗi đã sửa: trước đó `"auto"` lọt xuống
> `PersonTrackingConfig` làm mọi frame hành vi ném `Invalid CUDA 'device=auto'`.

### 5.1 Máy Windows + NVIDIA + WSL2

Hai luồng chạy được, nhưng có ba chỗ không tự nhiên mà có:

1. **File không nằm trong git** — phải chuyển kèm: `stage6_bundle_exact/` (18MB, XGBoost causal; thiếu thì
   `action` chỉ in `[WARNING] Không thể nạp Causal Action Classifier` rồi **im lặng 0 cảnh báo** —
   `.gitignore:88`), và `data/student_faces/` (thiếu thì mọi khuôn mặt đều là `NGUOI LA`; đăng ký lại bằng
   nút 📸). `backend/ai_services/whisper/weights/` (PhoWhisper bản OpenVINO) **không cần** trên máy NVIDIA:
   `phowhisper_service.py:16` thấy CUDA là tự tải `vinai/PhoWhisper-small` (cần internet lần đầu).
2. **Cài đúng bản CUDA** — `requirements.txt` đang ghi `onnxruntime` (CPU, dòng 22) và chú thích torch bản
   `cpu` (dòng 2); trên RTX phải đổi sang `onnxruntime-gpu` + torch `--index-url .../whl/cu12x`, ngược lại
   **không** cần 3 dòng `openvino/onnxruntime-openvino/optimum` (77-81). `sudo apt install ffmpeg
   portaudio19-dev libgl1` — ffmpeg bắt buộc để UI phát được video offline (`to_h264`).
3. **Camera thì ổn, micro là chỗ dễ gãy** — frame WebRTC đi từ trình duyệt phía Windows nên WSL không cần
   `/dev/video0`; nhưng micro được thu **phía server** (`audio: False` trong `media_stream_constraints`), mà
   PyAudio/PortAudio trong WSL chỉ thấy PulseAudio nếu có `pulseaudio` + `libasound2-plugins`. Micro hỏng thì
   app vẫn lên bình thường (lỗi được catch), chỉ mất cảnh báo âm thanh. Luồng **offline không cần micro**,
   nên đây là cách an toàn nhất để kiểm chứng mọi thứ trên WSL: `.venv/bin/python run_offline.py <file>`.

Export model cho máy chỉ có iGPU:

```bash
python scripts/export_models.py            # weights/best (1).pt + yolov8n.pt -> _openvino_model/ + .onnx
python scripts/check_env.py                # chẩn đoán CUDA / OpenVINO / driver
```

---

## 6. Cài đặt và chạy

```bash
cd Real-Time-Exam-Proctoring-System
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Ubuntu / Intel iGPU — các gói OpenVINO bị COMMENT ở requirements.txt:77-81, phải tự cài:
uv pip install openvino onnxruntime-openvino "optimum[openvino]"
sudo apt-get install portaudio19-dev        # cho pyaudio
```

Chạy app thật (không phải `uv run -m backend.main` — backend đã bị comment):

```bash
# Ubuntu / Intel iGPU
streamlit run app_webrtc.py

# Windows / NVIDIA (shim 12 dòng, exec đúng file ở trên — xem §5)
streamlit run app_webrtc_win.py

# Offline: 1 file video -> video burn cảnh báo + báo cáo (không cần micro/camera/WebRTC)
.venv/bin/python run_offline.py data/IMG_7817.MOV --stride-heavy 3 --stride-paper 5 \
    --stride-pose 3 --stride-action 3 --output offline_out/full
```

Mở `http://localhost:8501` → bấm **START** ở khối WebRTC → cấp quyền camera.
Micro được PyAudio/`sounddevice` chiếm trực tiếp tiến trình server, **không** cần cấp quyền trong
`media_stream_constraints` (`"audio": False`, `app_webrtc.py:517-524`).

Cấu hình thêm:

| Biến / tham số | Tác dụng |
| :-- | :-- |
| `PROCTOR_DEVICE=cuda\|intel\|cpu` | Ép thiết bị suy luận |
| `--server.headless true` | Chạy không mở browser tự động (tiện khi test bằng script) |
| `settings.face_similarity_threshold` (0.4) | Ngưỡng cosine → "người lạ" |
| `settings.object_detect_every_n_frames` (2) | stride YOLO giấy; **bị `FPS_SKIP=1` ghi đè ở luồng nặng** |
| `settings.paper_count_confirm_inferences` (2) | số lần inference liên tiếp để báo phao mới |

---

## 7. Vận hành giao diện

Cột trái: video WebRTC đã vẽ đè. Cột phải (`app_webrtc.py:530-556`):

1. **📸 Đăng ký khuôn mặt** — luồng nặng crop khuôn mặt lớn nhất (đã masking) +40px margin, lưu
   `data/student_faces/student_<ts>.jpg` (bản cho AI) và `data/registered_logs/full_student_<ts>.jpg`
   (bản toàn cảnh cho người), rồi `face_model._load_database()` nạp lại.
2. **🔒 Cố định Baseline (A)** / **🔄 Reset Setup (D)** — ARM/DISARM pipeline giấy; dòng `st.info` phía dưới
   hiện `paper_status_msg` ("SETUP…" / "ARMED! Baseline: N giấy.").
3. **Lịch sử cảnh báo** — `st_autorefresh(interval=3000)`, giữ 20 dòng gần nhất, có màu đỏ cho vi phạm.
4. **🎬 Chạy offline trên file video** (`app_webrtc.py:559-696`, khối expander cuối trang) — chọn nguồn file
   (đường dẫn trên máy, hoặc upload ≤200MB), chọn module + stride + `max-seconds` / `setup-seconds` / TTL rồi
   bấm Start. `OfflineJob` (`offline_job.py`) chạy `run_offline.main()` ở luồng nền; UI tự refresh mỗi 1.2s để
   hiện tiến độ + ảnh khung hình đang xử lý (nhịp 4s, `OFFLINE_REFRESH_MS`), xong thì **phát video kết quả ngay
   trong trang** (§11.4). Nút bị chặn khi camera realtime đang phát.

Overlay trên video: đỏ = vật cấm, vàng cam = `NGUOI LA`, cyan = giấy baseline, đỏ đậm = `CHEAT SHEET (NEW)`,
dải đỏ đáy khung hình = cảnh báo tư thế/hành vi kèm `student_XX: <HÀNH VI>`. `OVERLAY_TTL = 1.5s`.

---

## 8. Đã kiểm chứng bằng chạy thật

Môi trường: Ubuntu, Python 3.12.3 (`.venv`), Intel Core Ultra 7 155H + iGPU Arc (`openvino 2026.3.0`,
`onnxruntime-openvino 1.24.1`, `optimum 2.3.0`, `torch 2.13.0+cpu`).

**Khởi động app** (`streamlit run app_webrtc.py --server.headless true`):

- OpenVINO thấy `['CPU', 'GPU']` → `resolve_device()` bên trong `ObjectDetector` chọn
  `intel (Intel(R) Graphics [0x7d55])`.
- Cả 4 thread nạp xong, UI render đủ (START / Select Device / 3 nút cột phải / log "Hệ thống đang chạy…").
- `ObjectDetector device=intel` + `weights/best_openvino_model`; `face_verify` chạy `OpenVINOExecutionProvider`;
  `PoseGaze device=intel`; `ObjectDetectModule` của pipeline giấy chạy **CPU** với `weights/best (1).pt`
  (đúng như app khai báo `device="cpu"`); `yolov8n.pt` được tự đổi sang `yolov8n_openvino_model`.
- Cảnh báo từ model: *"This checkpoint has no test_paper class. Paper labels will be exposed as
  `paper_unknown`"* — checkpoint hiện tại không có lớp `test_paper`.
- PhoWhisper + PhoBERT nạp trên OpenVINO GPU. **Lần chạy đầu tiên file `openvino_model.xml` bị export lại**
  (`phobert_service.py:96-104`) mất ~2 phút; sau khi có cache, khởi tạo pipeline âm thanh chỉ còn **6.3s**.

**Suy luận trên frame thật** (`data/IMG_7817.MOV`, 1920×1080, 1 frame):

| Service | Kết quả | Độ trễ |
| :-- | :-- | :-- |
| `ObjectDetector` | `None` (không có vật cấm trong frame) | 2.98s (gồm warm-up iGPU) |
| `FaceVerifier` | 2 khuôn mặt; `verify_face` → `alert`, `similarity_score 0.312 < 0.4`, `unauthorized_bbox [1089,574,1201,701]` | 4.53s (gồm nạp model) |
| `PaperCountPipeline` | 2 người, 3 tờ `cheat_sheet` `status=baseline_paper`; `arm()` → `mode=count_only`, `baseline_count=3`, `confirmation_frames=2`; `alerts=[]` | 3.6s cho init + 6 frame |
| `PoseGazeDetector` | `None` (single frame, chưa qua warmup 4 frame) | 3.4s |

**Âm thanh** (`data/test_audio/*.m4a|mp3`, qua `AudioPipeline.process_audio`):

| File | `transcription` | `risk` | `rule_label` | `ai_label` |
| :-- | :-- | :-- | :-- | :-- |
| cheat_test_1 | "câu hai chọn gì." | Cheating (0.9975) | **Medium** | Cheating |
| cheat_test_10 | "thôi hỏi bạn kia bài này làm sao." | Cheating (0.9879) | Safe | Cheating |
| cheat_test_2 | "kéo vài xen tí." | Cheating (0.9975) | Safe | Cheating |
| cheat_test_3 | "nhắc tao cầu cuối." | Cheating (0.9975) | Safe | Cheating |

→ Xác nhận mục ⚠️ ở §3.5: `rule_label` không bao giờ mang giá trị `Suspicious`/`Cheating`, nên quyết định
hoàn toàn dựa vào PhoBERT. Thời gian xử lý một câu 1.3–2.1s trên iGPU.

**Luồng offline** (`run_offline.py data/IMG_7817.MOV --stride-heavy 3 --stride-paper 5 --stride-pose 3
--stride-action 3`, 503 frame / 16.8s video, **225.8s** xử lý):

| Module | Kết quả trên file | Thời gian |
| :-- | :-- | :-- |
| `object` (YOLO vật cấm) | 168 lần chạy, **0** alert (không có điện thoại/tai nghe trong 16.8s này) | 0.040s/frame |
| `face` (verify + masking) | 168 alert `unauthorized_person` (DB chỉ có 1 khuôn mặt đã đăng ký) | 0.345s/frame |
| `paper` (đếm giấy theo người) | ARM sau 5s: baseline **2 giấy**; frame `t=7.83s` thấy tờ thứ 3 `suspicious_new_paper` (vẫn là frame `info`), cảnh báo đầu tiên `t=8.17s`, **28 alert** `paper_count_above_baseline`, owner `track_id=1` → `student_01` | stride 5 (không bấm timing riêng) |
| `pose` (`PoseGazeDetector`) | 168 lần chạy, 0 alert (chưa vượt warmup/gate) | 0.324s/frame |
| `action` (`gaze_scan` + causal) | 2 actor `student_01`/`student_02`, 168 frame, `predicted_class` vẫn `c5` dù `c2_score` max **0.9776** (gate + ngưỡng calibration chặn) | 0.345s/frame |
| `audio` (track của file) | 34 chunk → 4 câu có tiếng → **1** câu có transcription (`risk=Normal`), gắn mốc `t=0.0s` | init 8.4s |

Lặp lại đúng lệnh trên cho cùng kết luận và cùng mốc giấy
(7.83s / 8.17s / 15.33s, 28 cảnh báo), chỉ khác điểm classifier (`c2` max dao động 0.95–0.98) và thời gian
~5% — MediaPipe/ultralytics không đảm bảo bit-identical giữa hai lần chạy.

`report.json` có `alerts_by_actor = {"student_01": {"paper_count": 32}}` (32 = cả 4 frame `info` đã
nhìn thấy tờ giấy lạ nhưng chưa đủ xác nhận, chỉ 28 trong số đó là cảnh báo — xem §9 mục 11) và mục `actors` ghi điểm max của
từng actor (kể cả khi kết luận là `c5`) — đúng yêu cầu metric khoá theo `(video, actor_id)`.
`annotated.mp4` (1920×1080, kèm soundtrack gốc do ffmpeg remux) đã burn đủ: khung cam `Track n | conf`,
skeleton Holistic, `NGUOI LA`, `Paper #n` (vàng) / `CHEAT SHEET (NEW)` (đỏ) và bảng `CANH BAO` góc phải
theo mốc `[mm:ss]` của video. Riêng dải đỏ hành vi (`overlays.py:63-68`) **không** nổ trên video này vì
classifier kết luận `c5` cho cả hai actor; nó được kiểm chứng bằng alert `pose_gaze` giả — vẽ ra đúng
dòng `[CANH BAO TU THE]: STUDENT_01: EXCHANGE (TRAO ĐỔI)` + dòng log tương ứng ở bảng góc phải.

**Chưa kiểm chứng được**: vòng lặp frame WebRTC thời gian thực. Trình duyệt điều khiển tự động bị
`NotAllowedError: Permission denied` khi xin camera. Cần mở `http://localhost:8501` bằng trình duyệt thật
và bấm START để test end-to-end (ARM baseline, đăng ký mặt, cảnh báo overlay). Sau khi tách `pipeline/`,
app vẫn khởi động đủ 4 luồng (`[INFO] ✅ Fast Gaze Thread … đã nạp xong`, `[DEBUG AUDIO] ✅ RealtimeAudioWorker`,
`[Pipeline] Khởi tạo hoàn tất trong 12.06s`) và UI render như trước; phần chưa chạy được chỉ là lúc có
camera thật — mà nay `run_offline.py` đã phủ đúng các hàm per-frame đó trên file thật.

---

## 9. Giới hạn đã biết (đọc trước khi sửa)

1. **Chồng lặp tính toán ở luồng gaze**: `PoseGazeDetector.process_frame()` tự chạy tracking + holistic +
   classifier, rồi `pipeline/gaze_scan.py:134` (luồng `fast_gaze_thread`, `app_webrtc.py:339`) **chạy lại
   đúng stack đó** với một bộ tracking/holistic/classifier thứ hai ⇒ 2 lần YOLO + 2 lần MediaPipe mỗi frame.
   `run_offline.py` cố tình giữ nguyên hành vi này để so sánh được với realtime (tắt bằng `--modules`).
2. **`FPS_SKIP = 1`** bắt YOLO vật cấm + verify mặt chạy mọi frame, trái với chủ trương
   `object_detect_every_n_frames = 2` trong config.
3. ~~`core, devices = initialize_openvino()`~~ **đã sửa khi gộp entry point**: hàm cũ trả về MỘT giá trị
   (`devices`) nhưng bị unpack hai nên chỉ chạy được khi OpenVINO thấy đúng 2 thiết bị (`core='CPU'`,
   `devices='GPU'` — cả hai sai bản chất); máy chỉ có CPU hoặc có thêm NPU sẽ `ValueError` ngay lúc import.
   Nay là `_openvino_devices()` (`app_webrtc.py:66-76`) gán vào `OPENVINO_DEVICES` (`:86`), chỉ chạy ở nhánh
   Ubuntu, và `@st.cache_resource` vẫn giữ như cũ.
4. **stderr bị bịt**: `os.dup2(devnull, 2)` (`app_webrtc.py:13`) và `sys.stderr = devnull` (`:27`) làm mất
   traceback thật của MediaPipe/ONNX/OpenVINO. Khi cần gỡ lỗi, comment hai dòng này trước.
   Lưu ý thêm: `print()` của app ra **stdout đang được block-buffer**, log thường hiện thành từng cụm muộn.
5. **`suspicious_activity` chưa chia c1/c4** bằng YOLO như contract yêu cầu (§3.4).
6. **Re-ID người bằng appearance fingerprint đang chết**: `UltralyticsPersonDetector` không bao giờ đặt
   `appearance_fingerprint` (`tracking/detectors.py:58-75`) nên nhánh re-ID trong `tracker.py` không kích hoạt;
   `student_id` mất khi restart app vì session mới luôn mang timestamp.
7. **Code trùng lặp / package kép**: `backend/ai_services/pose_gaze/tracking/` và
   `backend/ai_services/pose_gaze/pose_gaze/tracking/` là hai bản khác nhau; app dùng `detectors.py` của bản
   ngoài nhưng `webcam.py`/`manager.py` của bản trong. Hai file entry point từng là bản sao 880 dòng; nay đã
   gộp về một (`app_webrtc_win.py` chỉ còn là shim 12 dòng, §5).
8. **Phụ thuộc không khai báo**: `openvino`, `onnxruntime-openvino`, `optimum[openvino]` đang bị comment ở
   `requirements.txt:77-81` nhưng bắt buộc phải có trên máy iGPU. `faster_whisper`/`ctranslate2`
   (`requirements.txt:13,16`) thì không nơi nào import.
9. **Tài liệu chết**: `backend/main.py` (FastAPI), `backend/api/pose_gaze_routes.py`, `app_ubuntu.py`,
   `app_win.py`, `whisper/realtime_whisper.py` (import lỗi), `whisper/whisper_gui.py`,
   `whisper/websocket_*.py`, `ObjectDetector._extract_server_detections()` (`object_detect.py:213`),
   nhánh COCO fallback (`object_detect.py:839`).
10. **Offline chậm hơn realtime rất nhiều vì một luồng**: 16.8s video hết 225.8s với stride 3/5/3/3 trên
     iGPU Arc (`face`+`pose`+`action` mỗi thứ ~0.33s). Realtime chạy song song 4 thread và bỏ frame cũ;
     offline không bỏ frame nên không có gì để so về độ trễ. Muốn nhanh: tăng stride, hoặc `--modules`.
11. **`alerts_by_module` / `alerts_by_actor` đếm cả frame "info"**: pipeline giấy push kết quả **mọi frame**
     nó chạy (`status="info"` khi chưa vi phạm) để UI vẽ được `Paper #n` — y hệt realtime. Vì vậy 101 dòng
     `paper_count` trong bảng §8 chỉ có 28 dòng là cảnh báo thật, và `alerts_by_actor` (32 cho `student_01`)
     vẫn tính 4 frame `info` có tờ giấy lạ. Muốn đếm đúng cảnh báo: lọc `status=="alert"` trong `events.jsonl`.
     `log.txt` và bảng `CANH BAO` thì đã lọc sẵn vì frame `info` không có dòng log (198/271 sự kiện).
12. **ARM baseline offline mốc theo thời gian, không theo người bấm nút**: `--setup-seconds` (mặc định 5s)
     tới là ARM, nên video quay sẵn tờ giấy thứ 3 trong 5s đầu sẽ coi tờ đó là bình thường. Đây là hệ quả
     của quyết định "tự động sau N giây đầu"; chưa có cách đánh dấu mốc ARM bằng tương tác.
13. **`annotated.mp4` chỉ còn 1 lớp encode**: `cv2.VideoWriter` xuất mpeg4 (`mp4v`) rồi ffmpeg remux `-c:v copy`,
     nên chất lượng video kết quả thấp hơn bản gốc; emoji trong bảng `CANH BAO` bị loại bỏ vì
     `cv2.putText` không vẽ được emoji (`overlays.py:17`).

---

## 10. Kiểm thử

```bash
.venv/bin/python -m unittest discover -s backend/ai_services/object_detect/tests -p "test_*.py"
.venv/bin/python -m unittest discover -s backend/ai_services/pose_gaze/tracking/tests -p "test_*.py"
```

Kết quả **chạy thật trên máy Ubuntu ngày 2026-10-27**:

| Suite | Kết quả |
| :-- | :-- |
| `pose_gaze/tracking/tests` | `Ran 36 tests — OK` |
| `object_detect/tests` | `Ran 32 tests — FAILED (failures=4)` |

4 failure của `object_detect` đều là **test giữ hợp đồng cũ**, không phải lỗi runtime:

- `test_banned_items_match_server_contract` — mong `BANNED_ITEMS = [cheat_sheet, earphone, smartwatch, smartphone]`,
  nhưng code đã comment `cheat_sheet` (`object_detect.py:28`) vì giấy chuyển sang pipeline riêng.
- `test_contract_resize_filter_scale_and_suppress_logging`, `test_no_banned_detection_returns_none`,
  `test_person_roi_remaps_phone_and_preserves_owner_hint` — cùng nhóm hợp đồng cũ về resize/ROI.

→ `chay.txt` (viết cho máy Windows khác) ghi "22 tests OK / 33 tests OK" là **đã cũ**; tài liệu gốc còn tham
chiếu các video `data/smartphone.mp4`, `data/cheatsheet.mp4`, `data/v_c1_s1_s2.mov` **không có** trong repo này
(thư mục `data/` chỉ có `IMG_7817.MOV`, `test_audio/`, `student_faces/`).

Test độc lập từng module (mở cửa sổ OpenCV, cần chạy từ thư mục gốc):

```bash
python -m backend.ai_services.object_detect.object_detect_test --source 0 --model "weights/best (1).pt" --confidence 0.5
python -m backend.ai_services.object_detect.test_paper_tracking_webcam --source 0
python -m backend.ai_services.object_detect.test_video_scenarios <video> --setup-seconds 5 --frame-stride 3 --max-seconds 10
```

---

## 11. Luồng offline (`run_offline.py`)

Một file video vào → **video đã burn cảnh báo + bộ dữ liệu giám sát** ra, không cần camera, micro,
WebRTC hay Streamlit. Đây là cách nhanh nhất để xem hệ thống làm gì với một tình huống cụ thể, và cũng
là cách kiểm chứng `pipeline/` (§2.4) chạy đúng khi không có browser.

```bash
.venv/bin/python run_offline.py data/IMG_7817.MOV                      # cả file, mặc định realtime-like
.venv/bin/python run_offline.py data/IMG_7817.MOV --max-seconds 6 \
    --stride-heavy 3 --stride-paper 5 --stride-pose 3 --stride-action 3 \
    --output offline_out/smoke
.venv/bin/python run_offline.py clip.mp4 --modules object,face,paper   # bỏ hành vi + âm thanh
```

### 11.1 Sản phẩm trong `--output`

| File | Nội dung |
| :-- | :-- |
| `annotated.mp4` | video đã vẽ y hệt khung realtime: `Track n \| conf` + skeleton Holistic, `NGUOI LA`, `Paper #n`/`CHEAT SHEET (NEW)`, dải đỏ hành vi (chỉ khi có alert `pose_gaze`), và **bảng `CANH BAO` góc phải** burn 8 dòng log gần nhất (thứ tự mới nhất trên như sidebar, mốc `[mm:ss]` theo video). Kèm soundtrack gốc nếu máy có `ffmpeg`. |
| `events.jsonl` | 1 dòng = 1 alert: `t`, `frame`, `module`, `status`, `actor_id`, `log`, toàn bộ payload gốc. |
| `log.txt` | đúng những dòng log **mà sidebar realtime cũng hiện**, theo thứ tự thời gian (đã bỏ markdown/HTML). Frame `info` của pipeline giấy không có log nên không xuất hiện ở đây — vẫn còn nguyên trong `events.jsonl`. |
| `report.json` | `alerts_by_module`, `alerts_by_actor`, `actors` (số frame, điểm max `c2/c3/suspicious`, `final_class`, `evidence_score`, `first_flag_frame` của **từng actor**), transcription âm thanh, cấu hình đã dùng và `timings_s` từng module. |
| `tracking/` | `tracking_state.json` của session giấy (`storage_root`). |

### 11.2 Tham số đáng chú ý

| Tham số | Mặc định | Ý nghĩa |
| :-- | :-- | :-- |
| `--modules` | `object,face,paper,pose,action,audio` | bật/tắt từng luồng |
| `--stride-heavy/-paper/-pose/-action` | `1/5/1/1` | y hệt stride realtime (`1/5/1/1`, §2.2); tăng lên để chạy nhanh |
| `--setup-seconds` | `5` | tới mốc này thì tự ARM baseline giấy (realtime là nút bấm) |
| `--ttl` | `1.5` | `OVERLAY_TTL` tính theo **giây video**, nên overlay trên `annotated.mp4` biến mất đúng như ngoài UI |
| `--max-seconds` / `--max-frames` | `0` (hết file) | cắt ngắn để test |
| `--audio-noise-seconds` | `0` | bản Ubuntu bỏ 1.0s đầu làm noise profile; file đã sạch ồn nên mặc định 0 |
| `--device` | tự chọn | `cuda\|intel\|cpu`, ghi đè `PROCTOR_DEVICE` |
| `--object-model` / `--person-model` / `--faces-dir` / `--confidence` | `weights/best (1).pt` / `weights/yolov8n.pt` / `data/student_faces/` / `0.7` | model + DB khuôn mặt; **cùng ngưỡng với app** (§5), `.pt` được `resolve_model_path()` map sang OpenVINO nếu máy có bản export |
| `--no-sound` / `--quiet` | off | không remux audio / không in tiến độ |

### 11.3 Giống và khác realtime

Giống: **cùng một hàm phát hiện** (`pipeline/*`), cùng định dạng alert, cùng cách dựng log
(`events.to_log_line`), cùng TTL overlay, cùng bộ cắt câu RMS-VAD, cùng pipeline giấy + ARM baseline,
cùng stack causal cho hành vi.

Khác: chạy tuần tự một luồng; `module="system"` (ARM) do runner tự sinh; mốc thời gian là giây trong
video thay vì giờ đồng hồ; bảng lịch sử burn vào frame thay vì sidebar; không có nút đăng ký khuôn mặt
(DB đọc thẳng từ `--faces-dir`); audio giải mã bằng `av` rồi mới cắt câu nên biết trước toàn bộ và
"nổ" đúng lúc video chạy tới câu đó (`run_offline.py:270-272`).

### 11.4 Chạy ngay trong giao diện (`offline_job.py` + expander cuối `app_webrtc.py`)

`OfflineJob` gọi `run_offline.main(argv, progress=...)` trong một daemon thread và giữ trạng thái để luồng
UI đọc (`snapshot()`); module này **không import streamlit** nên test độc lập được bằng Python thuần.
`run_offline.main()` phát tiến độ qua `_emit()` (`run_offline.py:386`) với các mốc
`init → models → audio → video → frame… → encode → done`.

| Thứ UI hiển thị | Nguồn |
| :-- | :-- |
| "Frame x/y · N cảnh báo" | emit `stage="frame"` mỗi frame (`run_offline.py:465`) |
| Ảnh preview | `OfflineJob._write_preview()` ghi `offline_out/<run>/preview.jpg` (rộng 640px, tối đa 1 ảnh/3s) |
| Video phát trong trình duyệt | `run_offline.to_h264()` (`:485`) remux `annotated.mp4` → `playback_h264.mp4` (libx264, yuv420p, +faststart) |
| Chỉ số, bảng theo actor, nút tải | `report.json` / `log.txt` / `events.jsonl` của lần chạy |

Ràng buộc đã kiểm chứng:

- `annotated.mp4` là **mpeg4 Simple Profile** — `st.video` của trình duyệt không phát được. CLI vẫn giữ mp4v
  (không đổi sản phẩm đã kiểm chứng), chỉ UI tạo thêm bản H.264: **3.7s** cho 16.8s video 1080p, 23.5MB → 5.0MB.
- Chặn chạy khi camera realtime đang phát (đọc `webrtc_ctx.state.playing`), vì luồng offline nạp **bộ model
  riêng** (YOLO + FaceNet + paper + gaze + PhoWhisper/PhoBERT) nên chạy cả hai là nhân đôi RAM/iGPU.
- `st.file_uploader` bị Streamlit chặn 200MB; file lớn hơn phải dùng ô đường dẫn. File upload được ghi ra
  `offline_out/_uploads/<kích_thước>_<tên>` vì `run_offline` cần đường dẫn thật.
- **Nhịp tự làm mới tốn hơn bản thân việc suy luận.** Đo cùng workload (241 frame, stride 3/5/3/3, iGPU Arc):
  CLI chạy một mình `face=0.414s / pose=0.327s / action=0.340s`; trong UI với refresh **1.2s** thành
  `1.203 / 0.993 / 1.137` (~3.3 lần chậm, cả lần chạy 503 frame hết 703s); với refresh **4.0s** còn
  `0.353 / 0.165 / 0.198` — nhanh hơn cả CLI vì model đã nạp sẵn trong tiến trình. Mỗi lần Streamlit rerun
  giành GIL với luồng suy luận và dựng lại element tree, nên `OFFLINE_REFRESH_MS = 4000` (`app_webrtc.py:565`)
  và `PREVIEW_INTERVAL_S = 3.0` (`offline_job.py:23`) đi đôi nhau.
- Không có nút huỷ giữa chừng: đã bấm Start thì phải chờ xong (`--max-seconds` là cách cắt ngắn).

Giới hạn riêng của bản offline: §9 mục 10–13.
