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

---

## 2. Kiến trúc thực tế

### 2.1 Đường đi của một khung hình

```
Browser (getUserMedia)
   │  WebRTC SENDRECV, audio:False
   ▼
video_frame_callback()                 app_webrtc.py:690   (chạy trên event loop, PHẢI nhanh)
   │  1. av.VideoFrame -> numpy BGR
   │  2. xúc RESULT_QUEUE -> ACTIVE_OVERLAYS + SHARED_STATE["logs"]
   │  3. bơm frame copy vào 3 queue theo stride
   │  4. draw_warning_overlays() -> trả frame đã vẽ về browser
   ▼
4 thread AI  ──(kết quả dict)──►  RESULT_QUEUE (maxsize=100)
```

Frame luôn **bỏ frame cũ nhất** khi queue đầy (`maxsize=2` + `get_nowait()` trước khi `put`) — ưu tiên
độ trễ thấp hơn là không rớt frame.

### 2.2 Phân phối frame (`app_webrtc.py:796-812`)

| Queue | Stride | Việc làm |
| :-- | :-- | :-- |
| `GAZE_QUEUE` | **mọi frame** (`% 1`) | pose/gaze + tracking + holistic + causal classifier |
| `PAPER_QUEUE` | mỗi 5 frame | đếm giấy theo người |
| `FRAME_QUEUE` | mỗi `FPS_SKIP = 1` frame → **mọi frame** | YOLO vật cấm + ArcFace verify |

`FPS_SKIP = 1` (`app_webrtc.py:597`) đang ghi đè mất `settings.object_detect_every_n_frames = 2`
(`backend/core/config.py:44`) ở tầng ứng dụng — và cũng khác luôn `FPS_SKIP = 5` khai báo ở
`config.py:7` (hằng số trùng tên nhưng app không import). Đây là điểm nóng nhất về hiệu năng của hệ thống.

### 2.3 Bốn luồng AI (`init_system_resources`, `app_webrtc.py:106`)

Tất cả chạy trong `@st.cache_resource` → **chỉ khởi tạo một lần cho cả tiến trình**. Bốn `Event`
(`vision_ready`, `paper_ready`, `audio_ready`, `gaze_ready`) được `wait()` trước khi hàm trả về
(`app_webrtc.py:587-590`), nghĩa là **trang web bị block ở trạng thái "Running…" cho tới khi cả 4 model
nạp xong** (vài phút ở lần chạy đầu, xem §8).

| Luồng | Dòng | Nạp gì | Đẩy vào `RESULT_QUEUE` |
| :-- | :-- | :-- | :-- |
| `heavy_vision_thread` | `:138` | `ObjectDetector` + `FaceVerifier` | `module="object_detect"`, `module="face_verify"` |
| `paper_vision_thread` | `:307` | `UltralyticsPersonDetector` + `ObjectDetectModule` + `PaperCountPipeline` | `module="paper_count"` |
| `fast_gaze_thread` | `:404` | `PoseGazeDetector` + `PersonTrackingModule` + `HolisticLandmarkExtractor` + causal classifier | `module="pose_gaze"` |
| `audio_ai_thread` | `:524` | `RealtimeAudioWorker` (PyAudio/sounddevice + VAD + PhoWhisper + PhoBERT) | `module="audio_phobert_pipeline"` |

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
  không đúp báo với pipeline giấy (`app_webrtc.py:261-269`).

### 3.2 Người lạ — `FaceVerifier` (`backend/ai_services/face_verify/face_verify.py`)

- **Không phải FaceNet/face_recognition** như tài liệu cũ. Dùng **insightface** qua ONNX Runtime, nhưng
  **không** nạp `FaceAnalysis` (5 model) mà tự chọn đúng 2 file: `det_10g.onnx` (RetinaFace) và
  `w600k_r50.onnx` (ArcFace, 512-d) từ pack `buffalo_l` trong `~/.insightface/models` (`face_verify.py:78-110`).
- `db_path="data/student_faces/"`: **`person_id` = tên file không phần mở rộng** (`SV001.jpg` → `SV001`).
  `_load_database()` giữ khuôn mặt lớn nhất mỗi ảnh, cache `normed_embedding`, tự bật FAISS khi DB lớn.
- So khớp: **cosine similarity** (`_known_vectors @ query`), ngưỡng `settings.face_similarity_threshold = 0.4`
  (`backend/core/config.py:143`). Dưới ngưỡng → `status="alert"`, `message="unauthorized_person"`,
  `details={similarity_score, unauthorized_bbox}` (`face_verify.py:187-213`).
- Mẹo **masking** trong app (`app_webrtc.py:156-195`): vẽ ô đen lên mọi khuôn mặt khác rồi verify từng mặt,
  để `verify_face` (vốn chỉ báo mặt đầu tiên khớp kém) không "ăn gian" kết quả giữa nhiều người.
  Trả `unauthorized_bbox` gốc về cho UI vẽ khung vàng "NGUOI LA".
  ⚠️ Giá phải trả: mỗi frame gọi `_detect_faces` `1 + N` lần (`verify_face` tự detect lại bên trong).

### 3.3 Giấy thi / phao — `PaperCountPipeline` + `CountBasedPaperMonitor`

- `ObjectDetectModule` (`object_detect.py:353+`) chạy YOLO **mỗi 2 frame** và phát lại kết quả cũ với
  `inference_ran=False` (`object_detect.py:586-611`); xác nhận dị thường theo **3/5 frame trùng vị trí**
  (`object_detect.py:1419-1455`). Kết quả nhét vào khoá `object_result` của pipeline.
- `PaperCountPipeline.process_frame()` trả: `session_id, frame_id, timestamp_ms, people, papers,
  paper_count_state, paper_monitoring_armed, object_result, alerts, risk_score` (`paper_count_pipeline.py:201-215`).
- Máy trạng thái (`paper_count_monitor.py`):
  - **SETUP** (`armed=False`): gom tối đa 60 snapshot số giấy, baseline = **mode** của chuỗi đó (`:507`).
  - **ARM** (`arm()`, `:80`): đóng băng `baseline_count` + `baseline_clusters`. Nút "🔒 Cố định Baseline (A)"
    gọi `arm_paper_monitoring()`; "🔄 Reset Setup (D)" tạo `session_id` mới (`app_webrtc.py:350-360`).
  - **Gom cụm**: union-find nhập 2 box khi IoP(phần tử nhỏ) ≥ `0.35` **hoặc** khoảng cách tâm chuẩn hoá
    ≤ `0.70` (`paper_count_monitor.py:285-330`; giá trị lấy từ `config.py:123-124` qua
    `paper_count_pipeline.py:61-66`, **không** dùng default `0.90` của constructor) — tránh gộp nhầm 2 tờ sát nhau.
  - **Baseline theo từng người**: giấy gán cho người gần nhất (`_nearest_person`), rồi so
    `c_count > baseline_owner_counts[owner]` **và** so tổng toàn cục; cần
    `paper_count_confirm_inferences = 2` lần inference liên tiếp mới báo (`:152-218`).
    Đây là bản vá lỗi "đếm toàn cục" mà tài liệu cũ mô tả.
- Mỗi phần tử `papers`: `{observation_index, bbox_xyxy, confidence, class_names, source_box_count,
  status, owner_track_id, owner_person_id}`; `status ∈ {baseline_paper, suspicious_new_paper}`.
  UI vẽ đỏ `CHEAT SHEET (NEW)` / vàng `Paper #n` (`app_webrtc.py:667-681`).
- Cảnh báo: `{source:"paper_count", label:"cheat_sheet", reason:"paper_count_above_baseline",
  risk_score:1.0, bbox_xyxy, owner_track_id, owner_person_id}`; app **vứt** alert `smartphone_detected`
  phát sinh từ module này (`app_webrtc.py:388-390`).
- Bật tắt: `smartphone_fallback_enabled = False` (`config.py:45`) và app còn force-tắt trong pipeline giấy
  (`app_webrtc.py:326-332`) ⇒ nhánh COCO fallback (`object_detect.py:839`) là **code chết**.

### 3.4 Hành vi — tracking + Holistic + causal classifier

`PoseGazeDetector` (`pose_gaze_service.py:83-141`) **không tự tính góc Euler**; nó là bọc ngoài của
cùng một stack: YOLO person tracking → MediaPipe Holistic theo crop từng người → XGBoost causal.
Trả `None` hoặc `{module:"pose_gaze", status:"warning", timestamp, details:{action, actor_id, score}}`.

- **Tracking** (`backend/ai_services/pose_gaze/tracking/`): `track_id` (số, tạm) vs `student_id`
  (định danh gán bởi giám thị, `manager.assign_student()`). `packet = TrackPacket(session_id, frame_id,
  timestamp_ms, tracks)`; track có `bbox, confidence, age_frames, missed_frames, is_present, student_id`.
  App tự đặt `student_id = "student_{track_id:02d}"` (`app_webrtc.py:479-485`).
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

- Thu: `realtime_audio_ubuntu.py` dùng **PyAudio** (paFloat32, chunk 0.5s, 16kHz mono) và `suppress_alsa_logs()`
  dup fd 2 ra `/dev/null`; `realtime_audio_worker.py` (Windows) dùng **sounddevice** và tự né thiết bị
  "stereo mix/loopback". Hai bản **chung một** RMS-VAD chunking: MIN 2.0s, MAX 6.0s, đuôi im lặng 0.8s,
  ngưỡng 0.02, overlap 1.5s khi cắt cưỡng bức.
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
├── app_webrtc.py            # ENTRY POINT Ubuntu / Intel iGPU  (Streamlit + WebRTC)
├── app_webrtc_win.py        # ENTRY POINT Windows / NVIDIA
├── app_ubuntu.py            # LEGACY: cv2.VideoCapture + cửa sổ OpenCV, không WebRTC
├── app_win.py               # LEGACY: như trên cho Windows
├── backend/
│   ├── core/
│   │   ├── config.py        # class Settings (plain class) — ngưỡng, stride, class names
│   │   └── device.py        # resolve_device / resolve_model_path / make_ort_session
│   ├── main.py              # DEAD: FastAPI app đã bị comment (dòng 6-14)
│   ├── api/pose_gaze_routes.py   # DEAD: không được mount
│   └── ai_services/
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

Hai entry point **không** đối xử như nhau:

| | `app_webrtc.py` (Ubuntu) | `app_webrtc_win.py` (Windows) |
| :-- | :-- | :-- |
| Chọn thiết bị | hardcode `device="GPU"`, `device="cpu"`; **không** import `select_device()` ở app | `select_device()` ngay đầu file (`:62-64`) |
| YOLO vật cấm | `weights/best_openvino_model` (`:145`) | `weights/best (1).pt` (`:135`) |
| Paper / tracking | `device="cpu"` (`:313`, `:315`, `:417`) | `device="auto"` (`:303`, `:305`, `:407`) |
| Micro | `realtime_audio_ubuntu.py` (PyAudio) | `realtime_audio_worker.py` (sounddevice) |

Hai file là **bản sao gần như hoàn hảo** (~880 dòng, diff chỉ 7 chỗ). Sửa logic ở một file phải sửa file kia.

> Ghi chú: `device="GPU"` vẫn hoạt động vì `ObjectDetector` tự gọi `resolve_device()` — alias `"gpu"`
> được chuẩn hoá về `auto` (`device.py:96`).

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

# Windows / NVIDIA
streamlit run app_webrtc_win.py
```

Mở `http://localhost:8501` → bấm **START** ở khối WebRTC → cấp quyền camera.
Micro được PyAudio/`sounddevice` chiếm trực tiếp tiến trình server, **không** cần cấp quyền trong
`media_stream_constraints` (`"audio": False`, `app_webrtc.py:862`).

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

Cột trái: video WebRTC đã vẽ đè. Cột phải (`app_webrtc.py:868-891`):

1. **📸 Đăng ký khuôn mặt** — luồng nặng crop khuôn mặt lớn nhất (đã masking) +40px margin, lưu
   `data/student_faces/student_<ts>.jpg` (bản cho AI) và `data/registered_logs/full_student_<ts>.jpg`
   (bản toàn cảnh cho người), rồi `face_model._load_database()` nạp lại.
2. **🔒 Cố định Baseline (A)** / **🔄 Reset Setup (D)** — ARM/DISARM pipeline giấy; dòng `st.info` phía dưới
   hiện `paper_status_msg` ("SETUP…" / "ARMED! Baseline: N giấy.").
3. **Lịch sử cảnh báo** — `st_autorefresh(interval=3000)`, giữ 20 dòng gần nhất, có màu đỏ cho vi phạm.

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

**Chưa kiểm chứng được**: vòng lặp frame WebRTC thời gian thực. Trình duyệt điều khiển tự động bị
`NotAllowedError: Permission denied` khi xin camera. Cần mở `http://localhost:8501` bằng trình duyệt thật
và bấm START để test end-to-end (ARM baseline, đăng ký mặt, cảnh báo overlay).

---

## 9. Giới hạn đã biết (đọc trước khi sửa)

1. **Chồng lặp tính toán ở luồng gaze**: `PoseGazeDetector.process_frame()` tự chạy tracking + holistic +
   classifier, rồi `app_webrtc.py:474-518` **chạy lại đúng stack đó** với một bộ tracking/holistic/classifier
   thứ hai ⇒ 2 lần YOLO + 2 lần MediaPipe mỗi frame.
2. **`FPS_SKIP = 1`** bắt YOLO vật cấm + verify mặt chạy mọi frame, trái với chủ trương
   `object_detect_every_n_frames = 2` trong config.
3. **`core, devices = initialize_openvino()`** (`app_webrtc.py:74`) là bug tiềm ẩn: hàm chỉ trả về
   `devices` (một list). Unpack chỉ thành công khi OpenVINO thấy **đúng 2** thiết bị; máy chỉ có CPU, hoặc có
   thêm NPU, sẽ `ValueError` ngay khi import. Hai biến này cũng không được dùng ở đâu.
4. **stderr bị bịt**: `os.dup2(devnull, 2)` (`app_webrtc.py:13`) và `sys.stderr = devnull` (`:27`) làm mất
   traceback thật của MediaPipe/ONNX/OpenVINO. Khi cần gỡ lỗi, comment hai dòng này trước.
   Lưu ý thêm: `print()` của app ra **stdout đang được block-buffer**, log thường hiện thành từng cụm muộn.
5. **`suspicious_activity` chưa chia c1/c4** bằng YOLO như contract yêu cầu (§3.4).
6. **Re-ID người bằng appearance fingerprint đang chết**: `UltralyticsPersonDetector` không bao giờ đặt
   `appearance_fingerprint` (`tracking/detectors.py:58-75`) nên nhánh re-ID trong `tracker.py` không kích hoạt;
   `student_id` mất khi restart app vì session mới luôn mang timestamp.
7. **Code trùng lặp / package kép**: `backend/ai_services/pose_gaze/tracking/` và
   `backend/ai_services/pose_gaze/pose_gaze/tracking/` là hai bản khác nhau; app dùng `detectors.py` của bản
   ngoài nhưng `webcam.py`/`manager.py` của bản trong. `app_webrtc.py` và `app_webrtc_win.py` cũng là bản sao.
8. **Phụ thuộc không khai báo**: `openvino`, `onnxruntime-openvino`, `optimum[openvino]` đang bị comment ở
   `requirements.txt:77-81` nhưng bắt buộc phải có trên máy iGPU. `faster_whisper`/`ctranslate2`
   (`requirements.txt:13,16`) thì không nơi nào import.
9. **Tài liệu chết**: `backend/main.py` (FastAPI), `backend/api/pose_gaze_routes.py`, `app_ubuntu.py`,
   `app_win.py`, `whisper/realtime_whisper.py` (import lỗi), `whisper/whisper_gui.py`,
   `whisper/websocket_*.py`, `ObjectDetector._extract_server_detections()` (`object_detect.py:213`),
   nhánh COCO fallback (`object_detect.py:839`).

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
