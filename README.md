# Hệ thống giám sát phòng thi bằng AI (Exam Proctoring System)

> Tài liệu này mô tả **kiến trúc thực tế đang chạy trong code**, không mô tả thiết kế ước nguyện.
> Mọi nhận định dưới đây đều kèm đường dẫn `file:dòng` và đã được kiểm chứng bằng cách chạy thử
> bản Ubuntu trên máy Intel Core Ultra 7 155H (iGPU Arc) ngày 2026-10-27 — xem [§8](#8-đã-kiểm-chứng-bằng-chạy-thật).

---

## 0. Đọc nhanh (AI hoặc người mới — khỏi phải đo lại)

**Là gì:** một tiến trình Streamlit duy nhất giám sát phòng thi qua WebRTC; 4 module AI chạy ở 4 daemon
thread và nói chuyện bằng `queue.Queue`; micro bị PyAudio/`sounddevice` chiếm trực tiếp chứ không đi qua
WebRTC. Không còn FastAPI backend, không có `frontend/`. Riêng **MediaPipe Holistic chạy ở một tiến trình
con** (`pipeline/holistic_worker.py`, socket AF_UNIX + pickle) vì nó giữ GIL ~100 ms mỗi tick và làm hình
camera giật/trễ — xem §2.3 và §8.3.

| Muốn gì | Mở đâu |
| :-- | :-- |
| Hiểu đường đi của frame + stride | §2.1, §2.2 |
| Phần phát hiện dùng chung (realtime + offline + khối UI) | `backend/ai_services/pipeline/` — §2.4 |
| Chạy camera thật | §6 (`streamlit run app_webrtc.py`) |
| Chạy 1 file video → video burn cảnh báo | §11 (`run_offline.py`) hoặc khối 🎬 cuối trang — §11.4 |
| Chọn thiết bị / model | §5 (theo `select_device().kind`, **không** theo OS) |
| Máy Windows + NVIDIA + WSL2 cần thêm gì | §5.1 |
| Biết module nào đang chậm / rớt frame | hộp **🩺 Sức khoẻ module** trên UI (§7) hoặc `scripts/replay_realtime.py` (§8.2) |

**Số đã đo trên máy tham chiếu (Intel Core Ultra 7 155H + iGPU Arc, Ubuntu, Python 3.12.3,
openvino 2026.3.0) — dùng làm mốc, đừng tốn 4 phút đo lại:**

| Đại lượng | Giá trị đo được |
| :-- | :-- |
| Thời gian một lần chạy module (offline, stride 3, đo 2026-10-28) | vật cấm `0.035s` · mặt `0.155s` · hành vi `0.186s` (module `pose` đã bỏ — §9 mục 1) |
| Nạp model | app `11.7–13.5s` lần đầu, `5–8s` khi OS đã cache file model; riêng audio (VAD+PhoWhisper+PhoBERT) `8.4s` |
| Thông lượng offline | 16.8s video (503 frame) hết `104.1s` với stride 3/5/3/3 ⇒ **~6.2× thời lượng video** (trước tối ưu: `225.8s`) |
| Sức chứa realtime (replay 480 frame @30fps, **bật cả 4 module**) | callback p50 `4.5ms` (28.6fps vào) · mặt p50 `251ms` · YOLO `56ms` · giấy `587ms` · hành vi `275ms`; rớt frame: gaze `426/480`, vision `433/480`, giấy `60/96`. Camera thật vẫn chưa đo (§8) |
| Chạy lại phép đo | `scripts/replay_realtime.py <video>` — không cần camera, dựng lại đúng 4 luồng + queue `maxsize=2` (§8.2) |
| Video thứ hai (2026-10-28) | `data/WIN_20260802_12_05_12_Pro.mp4` 1920×1080 @30fps, 532 frame: offline **145.5s** (stride 3/5/3/3), `face` 0.244s · `action` 0.306s; có **28 alert `pose_gaze`** |
| Chi phí đổi sang H.264 cho UI | `3.7s`, 23.5MB → 5.0MB |
| Ngưỡng phải nhớ | `OVERLAY_TTL=1.5s` · `LOG_COOLDOWN=2s` (gộp dòng log trùng) · queue worker `maxsize=2` (rơi frame cũ nhất) · `RESULT_QUEUE maxsize=100` · sidebar giữ 20 dòng · bảng burn vào video giữ 8 dòng |
| Causal classifier | cửa sổ **24** landmark frame/actor, warmup **4** ⇒ tăng stride `action` làm **đổi điểm** c2/c3, không chỉ đổi tốc độ |
| Tính tái lập của điểm hành vi | **Không** tái lập giữa 2 lần chạy cùng code: đo 2026-10-28 hai lần liền cho `c2` 0.6908 vs 0.6952, `suspicious` 0.8584 vs 0.8149 ⇒ chỉ A/B bằng **quyết định/cảnh báo + thời gian**, đừng A/B bằng điểm |

**Máy thứ hai — AMD Ryzen 7 7435HS (16 thread) + RTX 4050 Laptop, Windows 11 + WSL2 Ubuntu 24.04,
Python 3.12.3, torch 2.14.1+cu130, onnxruntime-gpu 1.31.1, 11 GB RAM thấy từ WSL2 (đo 2026-10-28).**
Số dưới đây là của **máy này**, không trộn với bảng Intel ở trên:

| Đại lượng | Trước (720p, gaze mỗi frame, MediaPipe trong app) | Sau (960×540, gaze 8 fps, vision 10 fps, MediaPipe tiến trình riêng) |
| :-- | :-- | :-- |
| callback WebRTC (bơm frame + vẽ) | p50 `32.9` · p95 `83.0` · max `211.8` ms | p50 `13.3` · p95 `24.0` · max `44.6` ms |
| VP8 encode (aiortc) | p50 `24.1` · p95 `41.3` · max `209.1` ms | p50 `10.5` · p95 `19.3` · max `42.0` ms |
| Gaze (tracking + holistic + classifier) | `3.4 fps` · p50 `294` ms · 2 track | `4.0–4.7 fps` · p50 `225–260` ms · 2 track |
| FaceNet (ArcFace, `CUDAExecutionProvider`) | p50 `232` ms trên CPU → `28.7` ms trên CUDA | p50 `44.8` ms @ 640 px trong luồng 10 fps |
| YOLO vật cấm (`fp16` khi CUDA) | p50 `17.6` ms standalone | p50 `24.8` ms trong luồng 10 fps |
| GIL bị MediaPipe giữ mỗi tick 2 track | `100–113 ms` (cold `886 ms`) | `0` — chạy ở tiến trình con |
| `task_input_size` 256/320/384/512 | không đổi chi phí (`151–157 ms`) | giữ mặc định 512 |
| Vẽ overlay trong callback (gồm khung xương) | p50 `4.1` · p95 `41.5` · max `48.4` ms | p50 `2.2` · p95 `6.0` · max `18.9` ms (riêng `draw_results` 2 track: p50 `0.66` ms) |

Ngân sách 20 fps là **50 ms/frame**; hai cột trên đo cùng một harness (`scripts/replay_realtime.py
--encode`) trên cùng video 20 s (nguồn 1280×720: cột "trước" giữ 720p + `--no-isolate`, cột "sau" thêm
`--resize 960x540`).

**Bẫy đã trả giá — đọc trước khi sửa:**
- `annotated.mp4` là mpeg4 Simple Profile ⇒ trình duyệt không phát được; UI phải tạo `playback_h264.mp4` (§11.4).
- `st_autorefresh` 1.2s làm suy luận chậm **~3.3 lần** (face 0.35s → 1.20s); đã chốt `OFFLINE_REFRESH_MS = 4000` (§11.4).
  Khối **log + sức khoẻ module của trang realtime** thì đã đổi sang `st.fragment(run_every="1s")`: rerun cả
  trang mỗi 3 s làm component WebRTC dựng lại và hình khựng đúng nhịp đó (§11.4).
- **MediaPipe Tasks giữ GIL suốt lần suy luận** (đo: `100–113 ms`/tick 2 track, lần đầu `886 ms`). Callback
  WebRTC là thread Python nên bị chặn; `streamlit_webrtc/process.py::AsyncMediaProcessTrack.recv` khi đó trả
  lại **`_last_out_frame`** (frame CŨ) ⇒ hình vừa giật vừa trễ dần. Đó là lý do tồn tại
  `backend/ai_services/pipeline/holistic_worker.py` (tiến trình con) — đừng "tối ưu" bằng cách bỏ nó.
- **Micro trong WSL2** cần plugin ALSA `pulse` (gói `libasound2-plugins`, mặc định KHÔNG có trong ảnh WSL).
  Thiếu nó thì PyAudio vẫn liệt kê `default` nhưng mở stream là lỗi; cách không cần `sudo` ở §5.1.
- streamlit_webrtc 0.77 **không** có `ctx.running`; phải dùng `webrtc_ctx.state.playing`.
- `SuppressOutput` / `suppress_alsa_logs` dup2 fd 1&2 của **cả tiến trình** lúc nạp model ⇒ mất vài dòng log
  "✅ đã nạp xong" của thread khác; thêm `PYTHONUNBUFFERED=1` khi cần xem log.
- `alerts_by_module` / `alerts_by_actor` đếm cả frame `status="info"` của pipeline giấy (§9 mục 11).
- **Chỉ còn MỘT stack hành vi** (`pipeline/gaze_scan.py`). `PoseGazeDetector` đã bị bỏ khỏi runtime ngày
  2026-10-28 (§9 mục 1); nó chỉ còn trong `app_ubuntu.py`, `app_win.py`, `backend/main.py` — các file legacy
  **không phải entry point**.
- `face_verify` bản cũ (tô đen mặt khác rồi detect lại) cho embedding lệch hẳn so với detect một lần: trên
  `data/IMG_7817.MOV` (cả 2 actor đều đã đăng ký) nó bắn **51 cảnh báo giả**, bản detect-once bắn **41**
  và nhanh **2.68×** (§9 mục 14). Đừng "khôi phục masking" mà không đo lại.
- Video test `data/IMG_7817.MOV` không có điện thoại/tai nghe và cả hai actor đều bị kết `c5` dù
  `c2_score` lên tới 0.61–0.98 (thay đổi mỗi lần chạy) ⇒ **không có cảnh báo hành vi là chuyện bình thường**,
  đừng đi tìm lỗi wiring (§8). Video `data/WIN_20260802_12_05_12_Pro.mp4` thì **có** cảnh báo hành vi
  (28 `pose_gaze` cho `student_01`) — dùng nó khi cần test nhánh cảnh báo.
- **Job offline phải chạy ở tiến trình riêng.** Chạy trong tiến trình Streamlit (bản trước 2026-10-28) là
  nạp bộ model thứ hai vào cùng process đang giữ ~3GB → cạn RAM, `oom_kill_process` giết cả app/IDE (§11.4).
- **TTL overlay không liên quan tới RAM hay tốc độ**: `--ttl 0.5` và `1.5` cho cùng thời gian (131.1s vs
  130.0s, overlay tối đa 2). Muốn giảm RAM thì giảm module/bớt giây video, không phải chỉnh TTL.

**Không có trong git, phải chuyển tay khi clone:** `stage6_bundle_exact/` (18MB artifact XGBoost causal —
thiếu thì module `action` chỉ in một dòng `[WARNING]` rồi **im lặng 0 cảnh báo**), `data/student_faces/`,
`backend/ai_services/whisper/weights/` (bản OpenVINO; máy NVIDIA tự tải `vinai/PhoWhisper-small` từ HF),
`offline_out/`.

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
   │  WebRTC SENDRECV, audio:False, 960x540@20fps
   ▼
video_frame_callback()                 app_webrtc.py:467   (chạy trên event loop, PHẢI nhanh < 50 ms)
   │  1. av.VideoFrame -> numpy BGR
   │  2. xúc RESULT_QUEUE -> ACTIVE_OVERLAYS + SHARED_STATE["logs"]  (gộp dòng trùng bằng LOG_COOLDOWN)
   │  3. bơm frame (copy 1 lần/frame) vào 3 queue: gaze 8 fps · giấy 4 fps · vision 10 fps
   │  4. draw_warning_overlays() -> trả frame đã vẽ về browser -> VP8 encode (~11 ms @540p)
   ▼
4 thread AI  ──(kết quả dict)──►  RESULT_QUEUE (maxsize=100)

fast_gaze_thread ──socket AF_UNIX + pickle──► tiến trình con MediaPipe
    (RemoteHolistic)                            (nice 10, giữ 4 core cho app)
```

Frame luôn **bỏ frame cũ nhất** khi queue đầy (`maxsize=2` + `get_nowait()` trước khi `put`) — ưu tiên
độ trễ thấp hơn là không rớt frame. Mọi lần rớt được đếm vào `METRICS` và hiện ở bảng
"🩺 Sức khoẻ module" (cột **rớt**).

### 2.2 Phân phối frame (`app_webrtc.py:513-535`)

| Queue | Nhịp (đo trên đồng hồ thật, không phải đếm frame) | Việc làm |
| :-- | :-- | :-- |
| `GAZE_QUEUE` | **8 fps** — `GAZE_TARGET_FPS = 8.0` (`:430`), khớp `temporal_policy.json target_fps=8` | tracking + holistic + causal classifier |
| `PAPER_QUEUE` | mỗi 5 frame (≈ 4 fps ở 20 fps) | đếm giấy theo người |
| `FRAME_QUEUE` | **10 fps** — `VISION_TARGET_FPS = 10.0` (`:424`) | YOLO vật cấm + ArcFace verify |

Trước 2026-10-28 cả hai queue đầu đẩy **mọi frame** (`FPS_SKIP = 1`). Với `maxsize=2` thì 2/3 số tick bị
`get_nowait()` vứt đi ngay, chỉ tốn CPU/GPU và tranh GIL với callback. Cổng thời gian còn đúng một tác dụng
phụ có lợi: nhịp vào model **không đổi** khi máy chậm (đếm frame thì tụt theo), nên cửa sổ 24 frame của
causal classifier vẫn tương ứng ~3 s như lúc train.

Mỗi frame callback chỉ `img.copy()` **một lần** rồi chia sẻ cho các queue (bản cũ copy 3 lần ở full-res).
`FPS_SKIP` trong `backend/core/config.py` là hằng số legacy, app không import.

### 2.3 Bốn luồng AI (`init_system_resources`, `app_webrtc.py:118`)

Tất cả chạy trong `@st.cache_resource` → **chỉ khởi tạo một lần cho cả tiến trình**. Bốn `Event`
(`vision_ready`, `paper_ready`, `audio_ready`, `gaze_ready`) được `wait()` trước khi hàm trả về
(`app_webrtc.py:405-408`), nghĩa là **trang web bị block ở trạng thái "Running…" cho tới khi cả 4 model
nạp xong** (vài phút ở lần chạy đầu, xem §8).

| Luồng | Dòng | Nạp gì | Đẩy vào `RESULT_QUEUE` |
| :-- | :-- | :-- | :-- |
| `heavy_vision_thread` | `:151` | `ObjectDetector` + `FaceVerifier` | `module="object_detect"`, `module="face_verify"` |
| `paper_vision_thread` | `:238` | `UltralyticsPersonDetector` + `ObjectDetectModule` + `PaperCountPipeline` | `module="paper_count"` |
| `fast_gaze_thread` | `:323` | `PersonTrackingModule` + causal classifier (`gaze_scan.build_stack(isolate_holistic=True)`) trong app, **MediaPipe Holistic ở tiến trình con** (`pipeline/holistic_worker.RemoteHolistic`) — **stack hành vi duy nhất** | `module="pose_gaze"` |
| `audio_ai_thread` | `:358` | `RealtimeAudioWorker` (PyAudio/sounddevice + VAD + PhoWhisper + PhoBERT) | `module="audio_phobert_pipeline"` |

**Vì sao Holistic ở tiến trình riêng (§8.3 có số đo):** MediaPipe Tasks (pybind11) giữ GIL trong suốt lần
suy luận, ~100 ms cho 2 track. GIL là của **cả tiến trình**, nên thread callback + encode VP8 bị chặn; aiortc
hết frame kịp thì trả lại frame cũ → hình giật và trễ dần. Giao thức giữa hai bên:

```
cha -> con : {"cmd": "init", "kwargs": {...}}                    (một lần)
con -> cha : {"ok": True, "pid": ...}
cha -> con : {"cmd": "process", "frame": ndarray, "packet": TrackPacket}
con -> cha : {"ok": True, "results": tuple[TrackHolisticResult, ...]}
cha -> con : {"cmd": "quit"}
```

- Chỉ **dataclass thuần** (`TrackHolisticResult` / `LandmarkPoint` / `BoundingBox`) qua dây, không có protobuf.
- Tiến trình con chạy `nice 10` và bị giới hạn affinity còn `cpu_count - PROCTOR_HOLISTIC_RESERVED_CORES`
  core (mặc định để dành 4) — nếu để MediaPipe ăn hết 16 thread thì encode 720p từ 7 ms tăng lên 26 ms.
- Con chết/timeout (`request_timeout=4s`, `max_restarts=3`) → `RemoteHolistic` tự dựng lại; quá số lần thì
  **chạy Holistic ngay trong app** như cũ, luồng hành vi không bao giờ chết, chỉ chậm lại. Trạng thái hiện
  tại in ở hộp 🩺 (`mode=process|in-process`, `remote=`, `tại chỗ=`).
- Vẽ khung xương **không** đi qua tiến trình con: `draw_results` là hàm OpenCV thuần, chạy trên một
  `HolisticLandmarkExtractor` chưa nạp model ở tiến trình chính.
- Chạy thử riêng: `.venv/bin/python -m backend.ai_services.pipeline.holistic_worker --selftest`.
- Offline (`run_offline.py`) **không** tách tiến trình: chạy theo lô, không ai xem trực tiếp.

### 2.4 Package dùng chung cho mọi runtime: `backend/ai_services/pipeline/`

Phần logic "1 frame vào → alert ra" không còn nằm trong hai file app nữa. Mỗi module là hàm thuần,
**không thread, không queue, không `st.`**, nên realtime và offline gọi cùng một code:

| Module | Hàm | Realtime gọi ở | Offline gọi ở |
| :-- | :-- | :-- | :-- |
| `events.py` | `to_log_line` `:16`, `prune_overlays` `:78`, `expire_overlays` `:87` | `app_webrtc.py:469` / `:477` / `:425` | `run_offline.py:149` / `:166` / `:172` |
| `overlays.py` | `draw` `:26` (+ `log_lines` để burn cột lịch sử) | `app_webrtc.py:428` | `run_offline.py:173` |
| `face_scan.py` | `scan_stranger_with_masking` `:23` (detect 1 lần/ frame), `crop_for_registration` `:56` | `app_webrtc.py:171` / `:187` | `run_offline.py:276` |
| `object_scan.py` | `scan_objects` `:14` (lọc nhãn giấy khỏi vật cấm) | `app_webrtc.py:220` | `run_offline.py:274` |
| `paper_scan.py` | `arm_baseline` `:12`, `build_alert` `:23` | `app_webrtc.py:282` / `:311` | `run_offline.py:337` / `:347` |
| `gaze_scan.py` | `build_stack` `:113` (`isolate_holistic=`), `scan_frame` `:168` (trả **list** alert, mỗi actor một alert), `GazeStack.draw_skeleton` `:82` | `app_webrtc.py:330` / `:350` | `run_offline.py:269` / `:299` |
| `holistic_worker.py` | `RemoteHolistic` `:69` (proxy cùng giao diện `HolisticLandmarkExtractor`), `worker_main` `:299`, `--selftest` `:357` | `gaze_scan.build_stack` khi `isolate_holistic=True` (`app_webrtc.py:333`) | — (offline chạy tại chỗ) |
| `audio_scan.py` | `DynamicSpeechChunker` `:28`, `is_alert` `:121`, `to_alert` `:127` | `realtime_audio_ubuntu.py:49` + `:98`; `realtime_audio_worker.py:28` | `run_offline.py:97` (chunker) + `:120` (`to_alert`) |
| `metrics.py` | `METRICS.time(stage)` / `METRICS.drop(stage)` / `METRICS.snapshot()` | 8 stage trong app (§8.2) | `scripts/replay_realtime.py` |

Hai hàm cũ trong app giữ nguyên chữ ký nhưng chỉ delegate — `get_largest_stranger_with_masking`
(`app_webrtc.py:169-170`) và `draw_warning_overlays` (`app_webrtc.py:421-427`) — nên không phải sửa các chỗ gọi.
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
- `pipeline/face_scan.py:23` (`scan_stranger_with_masking`) detect **một lần** cho cả frame, rồi so khớp
  `normed_embedding` của từng mặt theo thứ tự diện tích giảm dần và trả người lạ đầu tiên gặp.
  Bản 2026-10-28 trước đây tô đen các mặt khác rồi gọi `verify_face` (tự detect lại) cho từng người;
  cách đó tốn `1 + N` lần detect/frame và — đo trên clip thật — cho embedding lệch, nhiều cảnh báo giả hơn
  (§9 mục 14). Tên hàm giữ nguyên để không phải sửa chỗ gọi.
  Cùng module này có `crop_for_registration()` (`:56`) dùng cho nút đăng ký mặt (+40px quanh khuôn lớn nhất).

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

Toàn bộ stack này nằm trong `pipeline/gaze_scan.py`: `build_stack()` (`:113-165`) nạp
`PersonTrackingModule` + `HolisticLandmarkExtractor` + classifier, `scan_frame()` (`:168-212`) chạy một
frame và **trả về danh sách alert, mỗi actor một alert** — nên realtime (`app_webrtc.py:330/350`) và
offline (`run_offline.py:269/299`) dùng chung hai lời gọi này. **Đây là stack hành vi duy nhất**:
`GazeStack` giữ `latest_packet`/`latest_results` và `draw_skeleton()` (`:82`) để UI vẽ khung xương trên
frame hiển thị mà không chạy model lần hai.

`build_stack(isolate_holistic=True)` (chỉ realtime dùng) thay `HolisticLandmarkExtractor` bằng
`RemoteHolistic` — proxy **cùng giao diện** nhưng `process_packet()` đi qua socket sang tiến trình con
(`pipeline/holistic_worker.py`, §2.3/§8.3). Nhờ vậy thời gian Holistic (trung bình `225–260 ms` cho 2 track
trên RTX 4050) không còn chặn callback WebRTC, dù gaze thực tế chỉ đạt **4.0–4.7 fps** chứ không phải 8 fps
như cổng thời gian đặt ra — xem §9.

`PoseGazeDetector` (`pose_gaze_service.py:83-141`) là vỏ ngoài cũ của cùng stack (YOLO person tracking →
MediaPipe Holistic theo crop từng người → XGBoost causal), trả
`{module:"pose_gaze", status:"warning", timestamp, details:{action, actor_id, score}}`. Từ 2026-10-28 nó
**không còn trong runtime** (chỉ còn trong `app_ubuntu.py`/`app_win.py`/`backend/main.py` — file legacy);
lý do bỏ và số đo ở §9 mục 1.

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
  Trong `pose_gaze_service.py:116-121` (file legacy, không còn chạy) vẫn còn nhãn `c1`/`c4` — **không bao
  giờ reachable**; điều kiện `"warming" not in str(classification)` (`:114`) cũng không lọc được gì vì
  không khoá nào chứa chuỗi đó (warmup thật do `predicted_class == "c5"` chặn). Runtime hiện tại chỉ phát
  `c2`/`c3`/`suspicious_activity` từ `gaze_scan.ALERT_CLASSES`.

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
│       │   ├── face_scan.py #   verify người lạ (detect 1 lần) + crop đăng ký mặt
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

   Bản đã kiểm trên RTX 4050 (2026-10-28): `torch 2.14.1+cu130` + **`onnxruntime-gpu==1.31.1`** (khớp CUDA 13
   và cuDNN 9 đi kèm torch; `pip uninstall -y onnxruntime` trước, hai gói này ghi đè cùng module `onnxruntime`).
   `scripts/check_env.py` phải in ra `CUDAExecutionProvider`. Đo được: `FaceVerifier._detect_faces` 1280×720
   **232 ms → 28.7 ms** chỉ nhờ đổi provider. `ObjectDetector` tự bật `half=True` khi `kind == "cuda"`.
   `face_verify.py:38-43` gọi `ort.preload_dlls()` để nạp cuDNN của torch, và vẫn rơi về CPU nếu CUDA lỗi.
3. **Camera thì ổn, micro là chỗ dễ gãy** — frame WebRTC đi từ trình duyệt phía Windows nên WSL không cần
   `/dev/video0`; nhưng micro được thu **phía server** (`audio: False` trong `media_stream_constraints`), mà
   PyAudio/PortAudio trong WSL chỉ thấy PulseAudio nếu có `pulseaudio` + `libasound2-plugins`. Micro hỏng thì
   app vẫn lên bình thường (lỗi được catch), chỉ mất cảnh báo âm thanh. Luồng **offline không cần micro**,
   nên đây là cách an toàn nhất để kiểm chứng mọi thứ trên WSL: `.venv/bin/python run_offline.py <file>`.

   **Cách không cần `sudo` (đã chạy được trên máy RTX 4050, 2026-10-28).** Ảnh WSL không có sẵn plugin
   `pulse`, nên `pcm.!default { type pulse }` báo `cannot open shared library`; PyAudio *vẫn* liệt kê
   `default` (ALSA null device) nhưng mở stream là lỗi ngay. Giải nén gói vào `$HOME`:

   ```bash
   cd /tmp && apt-get download libasound2-plugins        # không cần root
   dpkg-deb -x libasound2-plugins_*.deb root
   mkdir -p ~/.local/lib && cp -r root/usr/lib/x86_64-linux-gnu/alsa-lib ~/.local/lib/
   printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' > ~/.asoundrc
   ```

   `realtime_audio_ubuntu.py:11-30` tự đặt `ALSA_PLUGIN_DIR=~/.local/lib/alsa-lib` **trước** `import pyaudio`
   khi thấy file đó tồn tại (biến đã có sẵn thì tôn trọng), nên chỉ cần 4 lệnh trên là app dùng được micro.
   Bản "sạch" nếu có quyền root: `sudo apt install -y libasound2-plugins pulseaudio-utils`.
   Kiểm tra nhanh: mở stream `pyaudio` 44.1 kHz đọc 8000 sample phải trả `rms` khác 0 — im lặng thì kiểm
   *Settings > Privacy > Microphone* của Windows (cho phép desktop apps) trước khi nghi code.

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
| `PROCTOR_HOLISTIC_RESERVED_CORES` (4) | Số core **để dành cho app** khi MediaPipe chạy ở tiến trình con (§2.3) |
| `PROCTOR_MIC_DEVICE=<index>` | Chọn thiết bị vào (mặc định = thiết bị mặc định của OS; index in ra ở console) |
| `PROCTOR_VAD_THRESHOLD` (0.02) | Ngưỡng RMS cắt câu của **luồng realtime**; hạ xuống 0.012 nếu mic nhỏ (§7 mục 6) |
| `--server.headless true` | Chạy không mở browser tự động (tiện khi test bằng script) |
| `settings.face_similarity_threshold` (0.4) | Ngưỡng cosine → "người lạ" |
| `settings.object_detect_every_n_frames` (2) | stride YOLO giấy (luồng nặng của app dùng cổng 10 fps, không dùng biến này) |
| `settings.paper_count_confirm_inferences` (2) | số lần inference liên tiếp để báo phao mới |

`.streamlit/config.toml` đặt `fileWatcherType = "poll"` và blacklist các thư mục lớn (`offline_out/`,
`data/`, `weights`, `.venv/`). Không có file này, Streamlit 1.61 trên máy 16 core in
`Failed to schedule watch observer for path ...` / `event_based_path_watcher.py` mỗi lần lưu file và còn
tốn inotify cho các thư mục output. Đã kiểm chứng 2026-10-29: 0 lần xuất hiện lỗi sau khi thêm cấu hình.

---

## 7. Vận hành giao diện

Cột trái: video WebRTC đã vẽ đè. Cột phải (`app_webrtc.py:669-686`):

1. **📸 Đăng ký khuôn mặt** — luồng nặng crop khuôn mặt lớn nhất +40px margin, lưu
   `data/student_faces/student_<ts>.jpg` (bản cho AI) và `data/registered_logs/full_student_<ts>.jpg`
   (bản toàn cảnh cho người), rồi `face_model._load_database()` nạp lại.
2. **🔒 Cố định Baseline (A)** / **🔄 Reset Setup (D)** — ARM/DISARM pipeline giấy; dòng `st.info` phía dưới
   hiện `paper_status_msg` ("SETUP…" / "ARMED! Baseline: N giấy.").
3. **Lịch sử cảnh báo** — `st.fragment(run_every="1s")` (`app_webrtc.py:591`), giữ 20 dòng gần nhất, có màu
   đỏ cho vi phạm. (Trước 2026-10-28 là `st_autorefresh(interval=3000)` ⇒ rerun **cả trang** mỗi 3 s, dựng lại
   component WebRTC và làm hình khựng đúng nhịp đó — §8.3.)
4. **🎬 Chạy offline trên file video** (khối expander cuối trang) — chọn nguồn file
   (đường dẫn trên máy, hoặc upload ≤200MB), chọn module + stride + `max-seconds` / `setup-seconds` / TTL rồi
   bấm Start. `OfflineJob` (`offline_job.py`) chạy `run_offline.main()` ở luồng nền; UI tự refresh mỗi 4s
   (`OFFLINE_REFRESH_MS`) để hiện tiến độ + ảnh khung hình đang xử lý, xong thì **phát video kết quả ngay
   trong trang** (§11.4). Nút bị chặn khi camera realtime đang phát.
5. **🩺 Sức khoẻ module** (`app_webrtc.py:606-624`, cùng fragment 1 s với log) — bảng fps / p50 / p95 /
   max / số frame rớt của từng stage, lấy trực tiếp từ `pipeline/metrics.METRICS`; kèm dòng trạng thái
   MediaPipe (`mode=process|in-process`, `remote=`, `tại chỗ=`) và nút **Reset số đo**.
   Đây là nơi trả lời "module nào đang tụt" mà không phải cắm profiler; số liệu tích luỹ tới lần reset
   (hoặc tới khi tiến trình restart).
6. **🎙️ Dòng trạng thái micro** (`_render_mic_status`) — thiết bị đang dùng + sample rate, RMS hiện tại,
   RMS cao nhất từ lúc chạy, ngưỡng VAD, số chunk đã nhận và số câu đã cắt. Dưới đây là chẩn đoán khi
   "nói mà không thấy gì" (số đo trên laptop Intel Core Ultra 7 155H, 2026-10-29: nền phòng RMS ≈ 0.006,
   `Dmic0` 91% / -8 dB, phát lại file test ở 100% âm lượng thì cắt được câu, ở 25% thì không):

   | Hiện tượng | Nguyên nhân | Xử lý |
   | :-- | :-- | :-- |
   | `chunk = 0`, RMS = 0.0000 | không mở được thiết bị / thiết bị câm | console có `[Audio] ❌ Không mở được micro` + danh sách index; chạy lại với `PROCTOR_MIC_DEVICE=<index>` |
   | chunk tăng, RMS cao nhất < ngưỡng | mic quá nhỏ so với 0.02 | `PROCTOR_VAD_THRESHOLD=0.012` (chỉ ảnh hưởng realtime, không đổi offline) |
   | chunk tăng, "câu đã cắt" > 0, console có transcript mà panel trống | camera chưa START nên `_process_incoming_frame` không chạy ⇒ `RESULT_QUEUE` không được rút | bấm START camera, hoặc xem console |
   | "câu đã cắt" > 0 nhưng transcript rỗng | Silero-VAD (ngưỡng 0.65) hoặc PhoWhisper loại câu | nói to/rõ hơn hoặc hạ ngưỡng chunker như trên |

   Hai biến môi trường trên chỉ đọc lúc khởi động, phải chạy lại app. Danh sách thiết bị vào được in ở
   console mỗi lần start (và in đầy đủ khi mở stream thất bại).

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
| `PoseGazeDetector` | `None` (single frame, chưa qua warmup 4 frame) — **không còn trong runtime**, giữ làm mốc lịch sử | 3.4s |

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
--stride-action 3`, 503 frame / 16.8s video, **104.1s** xử lý — trước tối ưu 2026-10-28 là 225.8s):

| Module | Kết quả trên file | Thời gian |
| :-- | :-- | :-- |
| `object` (YOLO vật cấm) | 168 lần chạy, **0** alert (không có điện thoại/tai nghe trong 16.8s này) | 0.035s/frame |
| `face` (verify, detect 1 lần/frame) | **41** alert `unauthorized_person`. Cả 2 actor đều ĐÃ đăng ký (DB có 2 mặt) ⇒ đây đều là cảnh báo giả; bản masking cũ bắn 51 | 0.155s/frame |
| `paper` (đếm giấy theo người) | ARM sau 5s: baseline **2 giấy**; frame `t=7.83s` thấy tờ thứ 3 `suspicious_new_paper` (vẫn là frame `info`), cảnh báo đầu tiên `t=8.17s`, **28 alert** `paper_count_above_baseline`, owner `track_id=1` → `student_01` | stride 5 (không bấm timing riêng) |
| `pose` | **Đã bỏ.** `--modules pose` giờ là bí danh của `action` (một stack duy nhất), không còn khởi tạo `PoseGazeDetector` | — |
| `action` (`gaze_scan` + causal) | 2 actor `student_01`/`student_02`, 168 frame, `predicted_class` vẫn `c5` dù `c2_score` max **0.61** (gate + ngưỡng calibration chặn). Điểm **không tái lập** giữa 2 lần chạy (0.69 vs 0.70 ở lần đo riêng) | 0.186s/frame |
| `audio` (track của file) | 34 chunk → 4 câu có tiếng → **1** câu có transcription (`risk=Normal`), gắn mốc `t=0.0s` | init 6.0s |

Lặp lại đúng lệnh trên cho cùng kết luận và cùng mốc giấy
(7.83s / 8.17s / 15.33s, 28 cảnh báo), nhưng **điểm classifier không tái lập**: hai lần chạy liền nhau
(2026-10-28, cùng code, `--modules action`) cho `c2` 0.6908 vs 0.6952, `suspicious` 0.8584 vs 0.8149;
các lần chạy khác ngày còn lệch tới 0.61–0.98. Quyết định (`c5`, số alert) thì ổn định. Đừng dùng điểm
làm thước đo A/B — chỉ dùng quyết định/cảnh báo + thời gian.

### 8.2 Replay realtime không cần camera (`scripts/replay_realtime.py`)

Dựng lại đúng cấu trúc app (1 vòng bơm frame + 4 queue `maxsize=2` rớt frame cũ nhất + 4 luồng tiêu thụ),
phát video theo đúng fps để thấy fps thật và số frame rớt:

```bash
.venv/bin/python scripts/replay_realtime.py data/IMG_7817.MOV --seconds 16   # nhịp thật
.venv/bin/python scripts/replay_realtime.py data/IMG_7817.MOV --fast --modules action
# đo đúng thứ người dùng cảm nhận: encode VP8 như aiortc, hạ 540p, tách MediaPipe
.venv/bin/python scripts/replay_realtime.py <video> --seconds 15 --modules object,face,action \
    --encode --resize 960x540
# A/B cột "trước": để Holistic chạy tại chỗ trong tiến trình chính
.venv/bin/python scripts/replay_realtime.py <video> --encode --no-isolate
```

Cờ đáng nhớ: `--encode` (thêm stage `VP8 encode` giống aiortc), `--resize WxH`, `--no-isolate`
(`HolisticLandmarkExtractor` tại chỗ thay `RemoteHolistic`), `--json <file>` (ghi snapshot để so máy khác),
`--fast` (đọc video nhanh nhất để đo trần từng module).

Đo 2026-10-28, 16s @30fps, **bật cả 4 module**, cùng lúc trên iGPU (các module tranh nhau nên p50 cao hơn
chạy một mình):

| Stage | gọi | fps | p50 | p95 | rớt |
| :-- | --: | --: | --: | --: | --: |
| callback (bơm frame + vẽ) | 480 | 28.6 | 4.5ms | 23.9ms | — |
| vẽ overlay | 480 | 28.6 | 2.5ms | 15.3ms | — |
| dò vật cấm (YOLO) | 45 | 3.2 | 56ms | 77ms | vision 433/480 |
| dò mặt (detect-once) | 45 | 3.3 | 251ms | 342ms | vision 433/480 |
| đếm giấy | 34 | 2.4 | 587ms | 794ms | paper 60/96 |
| hành vi (tracking + causal) | 53 | 3.9 | 275ms | 348ms | gaze 426/480 |

Trước tối ưu cùng phép đo: mặt p50 708ms (2.8× chậm hơn), hành vi 2.3fps, gaze rớt 445/480. Bảng này cũng
chính là thứ hiện trong UI ở hộp **🩺 Sức khoẻ module** (thêm nút Reset số đo) — UI lấy số từ cùng registry
`pipeline/metrics.py`, nên không phải đo lại bằng mắt.

**Chưa kiểm chứng được**: vòng lặp frame WebRTC thời gian thực.

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

### 8.3 Tối ưu độ mượt realtime — máy RTX 4050 / WSL2 (2026-10-28)

Vấn đề người dùng báo: **hình camera bị trễ và giật**, không phải cảnh báo chậm. Cơ chế:

1. `streamlit_webrtc/process.py::AsyncMediaProcessTrack.recv` gọi `processor.recv(frame)` **tuần tự**; nếu
   callback chưa xong frame trước thì nó trả lại `_last_out_frame` (frame cũ) và bỏ frame mới.
2. Callback nằm cùng tiến trình với 4 luồng AI. MediaPipe Holistic giữ GIL ~100 ms/tick 2 track ⇒ trong lúc
   đó callback **không thể chạy**, nên cứ ~2 s lại có một khựng lớn, và hình tích luỹ trễ.

Sửa (đo bằng `scripts/replay_realtime.py --encode`, video 20 s, 4 module):

| Thay đổi | Vì sao |
| :-- | :-- |
| MediaPipe Holistic sang **tiến trình con** (`holistic_worker.py`) | GIL là của cả tiến trình; tách ra thì nó chậm bao nhiêu cũng không chặn video |
| `nice 10` + giữ lại 4 core (`PROCTOR_HOLISTIC_RESERVED_CORES`) | MediaPipe spawn thread theo số core; ăn hết 16 thread làm encode 720p 7 ms → 26 ms |
| Camera **960×540 @ 20 fps** (trước: `"video": True`, Chrome tự chọn, có lúc tụt 320×240) | VP8 encode 540p `11 ms` vs 720p `25 ms` khi máy bận; model chỉ dùng tối đa 640 px nên không mất chính xác, ảnh đăng ký mặt vẫn nét hơn 320×240 |
| Gaze **8 fps**, vision **10 fps** (trước: mọi frame) | `queue.maxsize=2` vứt 2/3 tick; 8 fps khớp `temporal_policy target_fps=8` của model causal |
| `img.copy()` **1 lần/frame** thay vì 3 | mỗi lần copy 720p là ~1 MB memcpy trong callback |
| `st.fragment(run_every="1s")` cho khối log/sức khoẻ (trước `st_autorefresh(3000)`) | rerun cả trang 3 s/lần ⇒ component WebRTC dựng lại + tranh GIL ⇒ hình khựng đúng nhịp 3 s |
| `cv2.polylines` gộp + gán numpy thay ~250 `cv2.line` + `cv2.circle`/track | mỗi lời gọi OpenCV là một round-trip GIL; p50 vẽ 2 track `0.66 ms` |
| `iceServers: []` (bỏ STUN Google) | client và server cùng máy; STUN chỉ thêm một vòng ICE ra Internet, treo nếu UDP bị chặn |
| `half=True` cho YOLO khi CUDA, `onnxruntime-gpu` cho ArcFace | 232 ms → 28.7 ms cho FaceNet |

Kết quả (cùng harness, cùng video; chi tiết ở bảng §0 "Máy thứ hai"):

| Stage | Trước | Sau |
| :-- | :-- | :-- |
| callback | p50 32.9 · p95 83.0 · **max 211.8** ms | p50 13.3 · p95 24.0 · **max 44.6** ms |
| VP8 encode | p50 24.1 · p95 41.3 · max 209.1 ms | p50 10.5 · p95 19.3 · max 42.0 ms |
| gaze | 3.4 fps (p50 294 ms) | 4.0–4.7 fps (p50 225–260 ms) |
| rớt frame `vision` | 118/300 | 16–19/300 |

Cách tắt khi cần so sánh: `scripts/replay_realtime.py --no-isolate` (chạy Holistic tại chỗ) và
`PROCTOR_HOLISTIC_RESERVED_CORES=0` (không giới hạn core của tiến trình con). Trên Windows bản native
`os.nice`/`sched_setaffinity` không tồn tại — code bỏ qua, chỉ mất phần ưu tiên CPU, vẫn tách tiến trình.

**Chưa đo được**: cảm nhận thật trên trình duyệt (cần camera + người dùng). Số trên là của harness phát lại
đúng cấu trúc callback + VP8 encode.

---

## 9. Giới hạn đã biết (đọc trước khi sửa)

1. ~~**Chồng lặp tính toán ở luồng gaze**~~ **đã sửa 2026-10-28**: trước đây `PoseGazeDetector.process_frame()`
   tự chạy tracking + holistic + classifier, rồi `gaze_scan.scan_frame()` **chạy lại đúng stack đó** với bộ
   thứ hai ⇒ 2 lần YOLO + 2 lần MediaPipe mỗi frame, và hai bộ alert `pose_gaze` khác nhau (bộ cũ còn bắn
   `c1`/`c4` trái contract). Nay chỉ còn `gaze_scan`; khung xương hiển thị được vẽ lại từ
   `GazeStack.latest_packet/latest_results` qua `GazeStack.draw_skeleton()` (`gaze_scan.py:82`), không chạy
   model lần hai. Offline: `--modules pose,action` chỉ chạy một lần. Đo được: offline 225.8s → 147.6s.
   **Hệ quả cần biết**: alert `pose_gaze` chỉ còn `c2`/`c3`/`suspicious_activity`, không còn nhãn
   "Sử dụng điện thoại/phao thi" do pose đoán (theo contract, c1/c4 phải do YOLO quyết).
2. ~~**`FPS_SKIP = 1`** bắt YOLO vật cấm + verify mặt chạy mọi frame~~ **đã sửa 2026-10-28**: thay bằng cổng
   thời gian `VISION_TARGET_FPS = 10` + `GAZE_TARGET_FPS = 8` (`app_webrtc.py:424-431`), vì `maxsize=2` vứt
   gần hết tick khi đẩy mọi frame (§2.2, §8.3).
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
10. **Offline chậm hơn realtime vì chạy tuần tự**: 16.8s video hết **104.1s** với stride 3/5/3/3 trên
     iGPU Arc (trước tối ưu: 225.8s). Realtime chạy song song 4 thread và bỏ frame cũ; offline không bỏ
     frame nên không có gì để so về độ trễ. Muốn nhanh: tăng stride, hoặc `--modules`.
     Ô nào là nút cổ chai thì xem bảng replay §8.2/UI thay vì đoán: hiện `paper` (587ms) là nặng nhất.
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
14. **`face_verify` detect một lần/frame (2026-10-28)**: bản cũ tô đen các mặt khác rồi detect lại cho từng
     người ⇒ N+1 lần RetinaFace + N(N-1) lần ArcFace mỗi frame (2 người = 3 detect + 4 embed). Nay
     `face_scan.scan_stranger_with_masking` chỉ detect 1 lần và so khớp `normed_embedding` có sẵn
     (`face_scan.py:23`). Đo trên `data/IMG_7817.MOV`: `0.290s → 0.108s`/frame (**2.68×**), 51 → 41 cảnh báo.
     **Đánh đổi đã đo**: quyết định lệch ở 16/160 frame so với bản cũ, vì detect trên ảnh bị tô đen làm
     landmark lệch nhẹ ⇒ embedding lệch (có frame tụt còn 0.02). Trên clip này cả 2 actor đều đã đăng ký nên
     mọi cảnh báo đều là giả; bản mới ít hơn. Không có người lạ thật trong clip ⇒ **chưa đo được recall**;
     cần clip có người lạ trước khi kết luận "chính xác hơn".
15. **Gộp cảnh báo trùng trên sidebar (`LOG_COOLDOWN = 2s`, `app_webrtc.py:416`)**: cùng module + cùng nội dung
     thì không chèn lại dòng log trong 2s. Overlay **vẫn** được làm mới mỗi alert, và `events.jsonl`/console
     vẫn ghi đủ từng lần phát hiện — con số "Log trùng đã gộp" nằm trong bảng 🩺. Đừng dùng bảng này để đếm
     cảnh báo; đếm bằng `report.json` / `events.jsonl`.
16. **OOM khi chạy offline trong tiến trình Streamlit (đã sửa 2026-10-28)**: app giữ ~3GB model realtime;
     job offline trong cùng process làm RSS ~4GB / virtual ~31GB / ~527 thread trên máy 15GB → `oom_kill_process`
     (syslog) giết app giữa lúc chạy. Nay `OfflineJob` chạy `run_offline.py` ở tiến trình con, đọc tiến độ
     `@@PROGRESS@@`, tự đặt `oom_score_adj=500`, có nút Huỷ, và UI chặn Start khi `MemAvailable < 2.0GB`.
     **Đừng đưa job về lại luồng nội bộ** dù thấy tiện — sẽ tái diễn OOM.
     Trong job con, thứ tự nạp là audio → phiên âm → `release_audio()` → thị giác: đỉnh 3.07GB → 2.66GB, và
     RSS tụt còn ~1.35GB trước khi chạy frame (đo trên video WIN_2026…).
17. **Không chia sẻ model giữa realtime và offline**: khác tiến trình thì không thể; cùng tiến trình thì
     tracker/MediaPipe/classifier/pipeline giấy đều có trạng thái riêng (track_id, cửa sổ 24 frame, session),
     dùng chung sẽ làm bẩn state realtime. Chỉ model thuần suy luận (YOLO vật cấm, FaceVerifier) là chia sẻ
     được, nhưng tiết kiệm không đáng so với rủi ro.
18. **Trần của gaze là ~4–5 fps, không phải 8 fps** (đo 2026-10-28, RTX 4050/WSL2, 2 track): một tick
     `process_packet` của MediaPipe mất `225–260 ms`, tức trần lý thuyết ~4 Hz. Cổng thời gian
     `GAZE_TARGET_FPS = 8` chỉ quyết định **nhịp đẩy** vào queue; queue `maxsize=2` rớt phần dư nên không có
     hại, nhưng đừng kỳ vọng 8 fps thật. Hạ `task_input_size` 512 → 256 **không** giúp (đo `150–157 ms` cho cả
     4 mức 256/320/384/512) — chi phí nằm ở số lần gọi model cho từng track, không ở pixel.
     Muốn nhanh hơn thật thì phải giảm số track (`max_tracks`), bỏ bớt landmark, hoặc dùng model nhẹ hơn —
     cả ba đều đổi phân bố đặc trưng của causal classifier, tức phải retrain, **đừng tự ý đổi**.
19. **Tiến trình Holistic và giới hạn core**: `PROCTOR_HOLISTIC_RESERVED_CORES` (mặc định 4) là số core
     **để dành cho app**. Đặt về 0 trên máy ít core (≤ 4) sẽ khiến encode VP8 chậm lại và hình giật trở lại —
     trên máy như vậy nên giảm `max_tracks` trước khi bỏ phần để dành core.

---

## 10. Kiểm thử

```bash
.venv/bin/python -m unittest discover -s backend/ai_services/object_detect/tests -p "test_*.py"
.venv/bin/python -m unittest discover -s backend/ai_services/pose_gaze/tracking/tests -p "test_*.py"
```

Kết quả **chạy thật trên máy Ubuntu** (2026-10-27, chạy lại 2026-10-28 sau các thay đổi P0/P1/P3: y hệt,
`object_detect.py` không bị sửa):

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
| `--modules` | `object,face,paper,pose,action,audio` | bật/tắt từng luồng. `pose` là **bí danh cũ của `action`** (một stack duy nhất) — chọn cả hai vẫn chạy một lần |
| `--stride-heavy/-paper/-pose/-action` | `1/5/1/1` | stride của offline (mặc định y như cũ). Realtime nay dùng **cổng thời gian** 10 fps (vision) / 8 fps (hành vi) / 4 fps (giấy), nên đừng so stride offline với realtime bằng số |
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

`OfflineJob` chạy `run_offline.py` ở **tiến trình con** (`subprocess.Popen`, `cwd` = repo root) và đọc tiến
độ từ stdout; module này **không import streamlit** nên test độc lập được bằng Python thuần.
Tiến trình con in mỗi sự kiện thành 1 dòng `@@PROGRESS@@{json}` (`run_offline._jsonl_progress`, bật bằng
`--progress-jsonl`) với các mốc `init → models → audio → video → frame… → encode → done`; các dòng log
thường được cha in lại ra console và giữ 25 dòng cuối để báo lỗi.

**Vì sao tiến trình con, không phải luồng nội bộ** (sự cố 2026-10-28): app realtime đã giữ ~3GB model.
Chạy offline trong cùng tiến trình là nạp thêm bộ thứ hai vào **một** process (RSS ~4GB, virtual ~31GB,
527 thread) → máy 15GB cạn swap và kernel bật OOM killer (`/var/log/syslog`: `oom_kill_process`), UI chết
giữa lúc chạy. Tách tiến trình: UI giữ nguyên ~3GB và phản hồi, job chết chỉ job mất, thoát là OS thu hồi
sạch; job con còn tự đặt `oom_score_adj=500` để nếu buộc phải giết thì giết job trước IDE/Streamlit.

| Thứ UI hiển thị | Nguồn |
| :-- | :-- |
| "Frame x/y · N cảnh báo" | emit `stage="frame"` mỗi frame (`run_offline.py:462`) |
| Ảnh preview | tiến trình con ghi `offline_out/<run>/preview.jpg` (rộng 640px, tối đa 1 ảnh/3s) khi có `--progress-jsonl`; cha chỉ đọc file |
| Video phát trong trình duyệt | `run_offline.to_h264()` (`:482`) remux `annotated.mp4` → `playback_h264.mp4` (libx264, yuv420p, +faststart) |
| Chỉ số, bảng theo actor, nút tải | `report.json` / `log.txt` / `events.jsonl` của lần chạy |
| Nút **⏹ Huỷ job** | `OfflineJob.cancel()` → SIGTERM, quá 5s thì SIGKILL; trạng thái `cancelled` |

Ràng buộc đã kiểm chứng:

- `annotated.mp4` là **mpeg4 Simple Profile** — `st.video` của trình duyệt không phát được. CLI vẫn giữ mp4v
  (không đổi sản phẩm đã kiểm chứng), chỉ UI tạo thêm bản H.264: **3.7s** cho 16.8s video 1080p, 23.5MB → 5.0MB.
- Chặn chạy khi camera realtime đang phát (đọc `webrtc_ctx.state.playing`), vì luồng offline nạp **bộ model
  riêng** (YOLO + FaceNet + paper + gaze + PhoWhisper/PhoBERT) nên chạy cả hai là nhân đôi RAM/iGPU.
- `st.file_uploader` bị Streamlit chặn 200MB; file lớn hơn phải dùng ô đường dẫn. File upload được ghi ra
  `offline_out/_uploads/<kích_thước>_<tên>` vì `run_offline` cần đường dẫn thật.
- **Nhịp tự làm mới tốn hơn bản thân việc suy luận.** Đo cùng workload (241 frame, stride 3/5/3/3, iGPU Arc):
  CLI chạy một mình `face=0.155s / action=0.186s` (2026-10-28); trong UI với refresh **1.2s** từng đo được
  `face 1.203 / action 1.137` (~3.3 lần chậm, cả lần chạy 503 frame hết 703s); với refresh **4.0s** còn
  `0.353 / 0.198` — nhanh hơn cả CLI vì model đã nạp sẵn trong tiến trình. Mỗi lần Streamlit rerun
  giành GIL với luồng suy luận và dựng lại element tree, nên `OFFLINE_REFRESH_MS = 4000` (`app_webrtc.py:695`)
  và `PREVIEW_INTERVAL_S = 3.0` (`offline_job.py:23`) đi đôi nhau. Lần chạy UI 8s ngày 2026-10-28 (241 frame,
  đủ 5 module): **69.1s**, tức 0.287s/frame — chậm hơn CLI ~40% vì lý do trên.
  *Khác trang realtime*: khối log + 🩺 của trang chính đã đổi sang `st.fragment(run_every="1s")`
  (`app_webrtc.py:591`) nên **không** rerun cả trang; khối offline vẫn phải `st_autorefresh(4000)` vì nó đọc
  file kết quả trên đĩa chứ không đọc biến trong RAM.
- **RAM là ràng buộc thật, không phải TTL.** Đo 2026-10-28 (CLI, cùng 8s, cùng stride): `--ttl 0.5` hết
  `131.1s`, `--ttl 1.5` hết `130.0s`, overlay tối đa 2 ở cả hai ⇒ TTL không làm tăng RAM/thời gian. Cái làm
  tràn RAM là tổng: app realtime ~3GB + job offline ~3GB + IDE/Chrome. Job con nay đỉnh ~2.7GB (trước 3.1GB)
  nhờ nạp audio → phiên âm → **giải phóng audio** → mới nạp thị giác; đo trên `WIN_20260802_12_05_12_Pro.mp4`:
  đỉnh 2.63GB ở bước audio rồi tụt còn 1.35GB trước khi vào phần video.
- UI đọc `/proc/meminfo` (`MemAvailable`): dưới **3.5GB** thì cảnh báo, dưới **2.0GB** thì **chặn nút Start**
  (chạy lúc đó chắc chắn OOM). Đóng bớt ứng dụng rồi tải lại trang.
- Không huỷ được job khi tiến trình cha chết đột ngột: `atexit` chỉ chạy khi Streamlit thoát bình thường.

Giới hạn riêng của bản offline: §9 mục 10–16.
