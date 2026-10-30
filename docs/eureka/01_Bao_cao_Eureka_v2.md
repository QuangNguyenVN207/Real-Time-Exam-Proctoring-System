# Giám sát phòng thi thời gian thực trên thiết bị biên
## Phụ lục B — Những gì MỚI so với bản báo cáo hội đồng đã đọc, và bằng chứng của chúng

> Tài liệu dành cho **team** chuẩn bị vòng chung kết (5 phút trình bày + phản biện).
> Nguyên tắc viết: **mọi con số đều kèm máy đo và lệnh tái lập**. Số nào chưa đo thì ghi rõ "chưa đo".
> Bản gốc hội đồng đã đọc: `data/Report_compressed.pdf` (mục lục 6 chương, Bảng 8–17).
> Quy ước: **[MỚI]** = xuất hiện sau bản PDF đó; **[CŨ]** = hội đồng đã thấy.

---

## 0. 60 giây đọc trước (đây là phần sẽ nói trên sân khấu)

**Bài toán của đề tài:** giám sát phòng thi bằng camera + micro, **ra quyết định theo từng thí sinh**
(actor-level), **chỉ dùng dữ liệu quá khứ** (nhân quả), **chạy thời gian thực trên máy tính bàn/laptop
không có cloud**.

**Bản cũ** chứng minh được *chất lượng mô hình*: macro-F1 hành vi 0.847 (grouped-OOF) / 0.767 (locked test),
face coverage 99.58%, PhoBERT F1 0.971, audio fusion F1 1.00 (n=40). Nhưng bản cũ **tự khai** ba chỗ trống
mà hội đồng chắc còn nghi ngờ:

1. §5.9: *"8 FPS là cấu hình lấy mẫu tầng pose, không phải FPS đã đo cho cả bốn mô-đun đồng thời"* — chưa có
   phiên chạy end-to-end mới.
2. §5.7: *"Thời gian CPU trung bình 15.69 giây/cặp, chưa đáp ứng xử lý tuần tự thời gian thực"* — audio chưa realtime.
3. §4.9: *"Việc có một lệnh khởi động thành công chưa đủ để xác nhận đang chạy mô hình thực"* — chưa đo hệ thống.

**Bản này đóng cả ba chỗ trống bằng số đo**, và đó chính là phần research: không phải "model tốt hơn",
mà là **phương pháp tổ chức hệ thống để nhiều mô hình AI cùng chạy thật trên thiết bị biên, và cách chứng minh
nó bằng đo lường**.

| # | Đóng góp (phương pháp) | Thay cho cách làm phổ biến | Kết quả đo được |
|:-:|:--|:--|:--|
| 1 | **[MỚI] Cách ly GIL bằng tiến trình con cho MediaPipe Holistic** (`pipeline/holistic_worker.py`) | Chạy mọi model trong cùng tiến trình Python (cách của bản cũ và của đa số demo Streamlit) | MediaPipe giữ GIL `100–113 ms`/tick → **0 ms**; callback WebRTC p50 `32.9→13.3 ms`, p95 `83.0→24.0 ms` (máy RTX 4050); trên máy iGPU Intel: `~31 ms/tick`, `remote=3 fallback=0` |
| 2 | **[MỚI] Lịch nhịp nhân quả (causal pacing)**: hành vi 8 fps (đúng `target_fps=8` lúc train), thị giác 10 fps, giấy mỗi 5 frame, queue `maxsize=2` drop-oldest | "Cứ mỗi frame chạy hết pipeline" (cách phổ biến) — vừa chậm vừa phát lại frame cũ | Thông lượng offline 16.8 s video: `225.8 s → 104.1 s` (**2.17×**); sức chứa realtime: callback p50 `4.5 ms` ở 28.6 fps vào |
| 3 | **[MỚI] Đặt mô hình đúng thiết bị + ngân sách thread đo được** (`select_device()`, `configure_cpu_threads()`, OpenVINO iGPU) | Mặc định ONNX/PyTorch chiếm hết 16–22 thread, mọi model giành nhau | Module giấy: `776 → 149 ms` trong app (**5.2×**), `1089 → 319 ms` ở harness; ablation thread YOLOv8n@1080p: 1→145 ms, 2→68 ms, 4→76 ms, 8→60 ms, **16→118 ms** (nhiều thread hơn *tệ hơn*); tổng app chỉ chiếm **2.5–3.2/22 core** |
| 4 | **[MỚI] Cổng chặn ảo giác ASR** (độ tự tin tiếng người + độ dài + chuẩn hoá mức) | Whisper luôn luôn trả văn bản → sinh câu bịa trên nhiễu, và pipeline cũ tin câu đó; cách chữa theo quán tính là "đổi model to hơn" | Trên input chỉ có nhiễu: câu bịa **2 → 0**, cảnh báo sai **2 → 0**; **12/12** file nói thật vẫn qua cổng (Bảng 3). Ở SNR +6 dB giữ nguyên văn 6/8 câu; ở −3 dB chỉ 2/8 qua cổng và **0/8 nguyên văn**, −6 dB cổng đóng **0/8** — hệ thống **im lặng thay vì bịa** (Bảng 4). Và đổi model **không** cứu được: F1 0.900 → 0.900 → **0.884** khi small → medium → large-v3, còn RTF tăng 0.56 → 3.71 (Bảng 2) |
| 5 | **[MỚI] Phát hiện lỗi hệ thống: "REMB giả nghẽn"** | Cho rằng hình mờ là do camera/độ phân giải, chỉ siết `getUserMedia` | aiortc ước lượng băng thông từ **jitter của chính tiến trình server**; khi server bận, nó báo Chrome hạ 960×540→640×360→320×180 ⇒ mờ hình. Sửa bằng **sàn REMB 1.2 Mbps** + log `[WEBRTC] đổi kích thước vào` để kiểm chứng |
| 6 | **[MỚI] Hợp đồng bằng chứng actor-level + harness tái lập** (`report.json` khoá `(video, actor_id)`, `scripts/replay_realtime.py`, `run_offline.py`) | Báo cáo metric theo cả video (không chỉ ra ai gian lận) | Mọi số liệu hành vi giờ truy vết được từng người, kể cả người bị kết `c5`; A/B không cần camera |

**Câu chốt để nói khi hết 5 phút:** *"Chúng tôi không thắng bằng mô hình to hơn. Chúng tôi thắng bằng cách
tổ chức hệ thống để bốn mô hình chạy thật trên một chiếc laptop — và chứng minh điều đó bằng số đo, kể cả
những số chỉ ra giới hạn của chính mình."*

---

## 1. Bảng delta chi tiết so với bản PDF hội đồng đã đọc

| Vị trí trong bản cũ | Hội đồng đã đọc thế | Bản này nói gì | Số mới |
|:--|:--|:--|:--|
| §5.1 Bảng 8, dòng "Hành vi OOF/locked" | macro-F1 0.84699 / 0.76735 | **Giữ nguyên, ghi rõ là số của bundle cũ** — không phải thứ chúng tôi cải thiện lần này | [CŨ] |
| §5.9 "Khả năng xử lý trực tiếp" | *chưa đo end-to-end mới* | Đã đo: 4 module đồng thời, có telemetry từng luồng, trên 2 máy khác nhau | p50/p95 callback, fps từng module, số frame rớt |
| §5.7 audio "15.69 s/cặp CPU" | chưa realtime | PhoWhisper-small trên **iGPU OpenVINO**: p50 **1.288 s/câu**, RTF **0.555** — không còn là bước chặn realtime. (Không quy đổi thẳng thành "nhanh hơn X×": 15.69 s là cho **một cặp** trên **CPU**, ≈7.8 s/câu ⇒ tốc độ tương đương **~6×**, và khác máy đo.) | `offline_out/asr_ablation/asr_small.json` |
| §5.7 "F1 = 1.000, n = 40, tập nội bộ dễ" | chính nhóm tự nhận là dễ | Chạy lại **trên cùng bộ 43 file có nhãn** (22 gian lận / 21 bình thường) với **3 cỡ model** + ablation cổng ảo giác | Bảng 2, Bảng 3 |
| §4.9 "không nên hiểu mặc định micro của trình duyệt đã về pipeline" | đúng, và vẫn đúng | Nói rõ kiến trúc: micro thu ở máy chạy ứng dụng; camera đi qua WebRTC; **hai đường độc lập** | — |
| §4.8.3 "runtime mặc định chỉ đọc thư mục cục bộ, không tự tải latest" | đúng | Vẫn vậy; nhưng đã thêm được so sánh medium/large-v3 để **trả lời câu hỏi 'sao không dùng model to hơn'** | Bảng 2 |
| Chương 2–4 (kiến trúc, 4 mô-đun AI) | mô tả bằng lời | Bổ sung **sơ đồ luồng + ngân sách thời gian** (50 ms/frame @20 fps, ai giữ thread nào) | §2 dưới đây |
| §5.8 face P1 "đánh giá nội bộ thầm dò" | đúng | Không đổi dữ liệu, nhưng đã thêm **quản lý danh tính trên UI** (đăng ký/xoá/sửa ảnh, nạp lại FAISS không cần khởi động lại) | demo |

---

## 2. Phương pháp — bốn kỹ thuật, mỗi cái là một quyết định thiết kế có đo

### 2.1 [MỚI] Cách ly GIL: vì sao "đa luồng Python" không phải là song song

**Vấn đề (mà các bài khác thường bỏ qua):** Python chỉ chạy được **một** đoạn bytecode mỗi lần. MediaPipe
Tasks giữ GIL **suốt** lượt suy luận — đo được `100–113 ms` mỗi tick 2 khuôn mặt (lần đầu `886 ms`).
Callback WebRTC cũng là thread Python ⇒ bị chặn ⇒ `streamlit_webrtc` phát lại **frame cũ**
(`AsyncMediaProcessTrack.recv` trả `_last_out_frame`). Hệ quả nhìn thấy được: hình vừa giật vừa trễ dần.

**Cách giải quyết:** đưa Holistic ra **tiến trình con** (`holistic_worker.py`), nói chuyện bằng pickle qua
socket AF_UNIX (Windows: named pipe), `nice 10`, dành 4 core, tự dựng lại tối đa 3 lần rồi mới rơi về
in-process (sống sót nhưng chậm).

**Kết quả đo:**

| Đại lượng | Trong tiến trình (cũ) | Tiến trình con (mới) | Máy |
|:--|:--|:--|:--|
| GIL bị giữ mỗi tick | 100–113 ms (cold 886 ms) | **0 ms** | RTX 4050 / WSL2 |
| callback WebRTC p50 / p95 | 32.9 / 83.0 ms | **13.3 / 24.0 ms** | RTX 4050 |
| VP8 encode p50 / p95 | 24.1 / 41.3 ms | 10.5 / 19.3 ms | RTX 4050 |
| Vẽ overlay p50 / p95 | 4.1 / 41.5 ms | 2.2 / **6.0 ms** | RTX 4050 |
| Nhịp Holistic | — | tick đầu 218.7 ms, sau đó **~31 ms/tick**, `remote=3 fallback=0` | iGPU Intel |

**Tái lập:** `.venv/bin/python -m backend.ai_services.pipeline.holistic_worker --selftest`
và `scripts/replay_realtime.py <video> --encode` (cột "trước" dùng `--no-isolate`).

> Điểm research: đây là **lỗi kiến trúc ngôn ngữ**, không phải lỗi mô hình. Nó giải thích vì sao hai hệ thống
> dùng *cùng model* lại cho trải nghiệm khác hẳn nhau.

### 2.2 [MỚI] Lịch nhịp nhân quả: chạy *đúng* tốc độ mô hình cần, không phải tốc độ camera

**Vấn đề:** camera vào 20–30 fps, nhưng (a) mô hình hành vi được train ở `target_fps=8`
(`stage6_bundle_exact/.../temporal_policy.json`), (b) khung hình tại thời điểm `t` **chưa tồn tại** cho các
frame tương lai ⇒ phải mask tương lai. Đẩy mọi frame vào mọi mô hình chỉ tạo ra: queue tràn, rơi frame,
tranh GIL với callback.

**Cách giải quyết:** mỗi luồng có **đồng hồ riêng** — hành vi 8 fps, thị giác 10 fps, giấy mỗi 5 frame,
micro không theo fps; queue `maxsize=2` **drop-oldest** (thà bỏ frame cũ nhất còn hơn trả lời bằng dữ liệu
cũ); cửa sổ landmark 24 frame/người + warmup 4.

**Kết quả đo:**

| Phép đo | Trước | Sau | Ghi chú |
|:--|:--|:--|:--|
| Offline 16.8 s video (503 frame), 4 module | 225.8 s | **104.1 s** | stride 3/5/3/3, máy iGPU |
| Callback khi replay 480 frame @30 fps, bật cả 4 module | — | p50 **4.5 ms** | harness không cần camera |
| Tầng hành vi | 3.4 fps | 4.0–4.7 fps | RTX 4050, 2 track |

**Bẫy đã trả giá (nói được khi phản biện):** tăng stride của `action` **làm đổi điểm** c2/c3 chứ không chỉ
đổi tốc độ, vì cửa sổ 24 frame bị co giãn. Nên chúng tôi **chốt 8 fps = đúng phân bố lúc train**, không
"tăng cho mượt".

### 2.3 [MỚI] Ngân sách tính toán: đo trước khi chia

**Vấn đề:** ONNX Runtime và PyTorch mặc định chiếm **toàn bộ** thread của máy. Bốn mô-đun cùng nạp ⇒
22 thread giành nhau, mỗi cái chậm đi, và RAM đỉnh tăng vì mỗi bộ tự giữ vùng nhớ lớn.

**Cách giải quyết:** `select_device().kind` (`cuda`/`intel`/`cpu`) chọn *model + thiết bị* theo **phần cứng**,
không theo OS; `configure_cpu_threads()` đặt `torch.set_num_threads`, `set_num_interop_threads(1)` và
`SessionOptions(intra_op=CPU_THREADS, inter_op=1)` cho **mọi** phiên ONNX Runtime; job offline chạy ở
**tiến trình riêng** với `oom_score_adj=500` để nó chết trước UI khi cạn RAM.

**Kết quả đo (ablation thread, YOLOv8n @1920×1080, máy iGPU Intel):**

| Số thread | 1 | 2 | 4 | 8 | 16 |
|:--|:--|:--|:--|:--|:--|
| Thời gian suy luận | 145 ms | 68 ms | 76 ms | 60 ms | **118 ms** |

⇒ Đường cong **không** đơn điệu: từ 8 thread trở lên bộ nhớ đệm/băng thông suy giảm hiệu năng. Đây là lý do
chúng tôi đặt `PROCTOR_CPU_THREADS=4` thay vì "mặc định".

| Phép đổi chỗ mô hình | Trước | Sau | Tốc độ |
|:--|:--|:--|:--|
| Detector giấy: CPU → iGPU OpenVINO (trong app) | 776 ms | **149 ms** | **5.2×** |
| Detector giấy: CPU → iGPU (harness replay) | 1089 ms | 319 ms | 3.4× |
| FaceNet ArcFace: CPU → CUDA (máy RTX 4050) | 232 ms | 28.7 ms | 8.1× |
| **Tổng CPU của cả app khi chạy 4 module** | — | **2.5–3.2 / 22 core** | còn dư cho 30 fps |

### 2.4 [MỚI] Cổng chặn ảo giác cho ASR — biến "model hay bịa" thành hệ thống đáng tin

**Vấn đề:** Whisper (mọi cỡ) **luôn sinh câu** khi gặp nhiễu. Bản cũ đã ghi nhận giới hạn này
(§4.8.3: *"Lỗi nhận dạng có thể làm mất hoặc thay đổi từ khoá"*) nhưng chưa có cơ chế chặn. Trên dữ liệu
thật của nhóm, một đoạn nhiễu 0.75 s sinh ra `"là tay bà mẹ đứng về."` và bị phân loại **Cheating** —
đó là cảnh báo sai *hệ thống*, không phải sai mô hình.

**Cách giải quyết (3 lớp, đều đo được):**
1. **Chuẩn hoá mức tự động** cho file video offline (RMS đích 0.05, chặn đỉnh 0.95, không bao giờ giảm) —
   `run_offline.py:_auto_gain`, cờ `--audio-gain`.
2. **Chuẩn hoá mức cục bộ** đoạn tiếng trước khi vào ASR (RMS đích 0.1) — `audio_pipeline._normalize_level`.
3. **Cổng độ tự tin**: chỉ đưa vào Whisper khi đoạn tiếng ≥ 0.8 s **và** độ tự tin tiếng người trung bình
   (Silero, tính trên chính đoạn đã cắt) ≥ 0.5. Phạm vi cần nói cho đúng: cổng **không chặn oan** tiếng người
   rõ (12/12 file nói thật đi qua, Bảng 3), nhưng tiếng người **lẫn sâu trong noise thì cũng bị chặn** —
   đo ở Bảng 4 (−3 dB: 2/8 qua cổng, −6 dB: 0/8). Đó là đánh đổi có chủ đích: thà mất một cảnh báo còn hơn
   ghi vào biên bản một câu không ai nói.

**Kết quả đo:** xem Bảng 3 (ảo giác) và Bảng 4 (robustness theo SNR).

### 2.5 [MỚI] "REMB giả nghẽn" — một lỗi chỉ xuất hiện khi hệ thống *của bạn* chậm

Quan sát: camera "lúc rõ lúc mờ". Đo trong app: `[WEBRTC] đổi kích thước vào: 960x540 -> 640x360` lặp lại
trong cùng một phiên. Nguyên nhân: aiortc ước lượng băng thông Chrome→server từ **jitter thời điểm gói RTP
đến tay tiến trình Python**; khi Python bận (GIL, bốn luồng AI), gói đến dồn cục ⇒ estimator báo nghẽn ⇒
Chrome bật `scaleResolutionDownBy` ⇒ **mờ hình, không phải do camera**.

Sửa: **sàn REMB** `PROCTOR_REMB_FLOOR=1.2 Mbps` (patch `RemoteBitrateEstimator.add`) + nâng `MIN_BITRATE`
VP8 `250 kbps → 1.2 Mbps`. Kiểm chứng bằng chính log đổi kích thước.

> Điểm research: đây là **vòng phản hồi điều khiển giữa tài nguyên CPU và chất lượng hình ảnh** — thứ không
> tồn tại trong pipeline offline, và rất ít bài proctoring nào nhắc tới.

---

## 3. KẾT QUẢ MỚI — đã chạy xong (gốc: `offline_out/asr_ablation/*.json`, `offline_out/bench_*/report.json`)

### Bảng 2 — Cùng bộ phân loại, đổi cỡ mô hình ASR (43 file có nhãn: 22 gian lận / 21 bình thường, máy M1)

| Model ASR | Thiết bị | P | R | F1 | p50 (s/câu) | p95 | RTF | RAM đỉnh (MB) |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| **PhoWhisper-small** (đang triển khai) | OpenVINO iGPU | **1.000** | 0.818 | **0.900** | **1.288** | 1.848 | **0.555** | **2094** |
| PhoWhisper-medium | CTranslate2 CPU int8 | 1.000 | 0.818 | 0.900 | 4.022 | 4.663 | 1.767 | 3106 |
| whisper-large-v3 | CTranslate2 CPU int8 | 0.905 | 0.864 | 0.884 | 8.555 | 9.888 | 3.706 | 4212 |

**Đọc bảng:** F1 **đi ngang rồi giảm** khi mô hình to lên, còn RTF tăng 6.6 lần và RAM tăng 2 lần.
Trên máy 15 GB không card rời, chỉ small + iGPU là chạy được realtime — đây là bằng chứng trực tiếp
cho quyết định kiến trúc, không phải tuyên bố.

*Tham chiếu chéo (không có văn bản chuẩn nên đo mức đồng thuận):* small–large khớp 46.5% số câu
(CER 0.0916, WER 0.1677); medium–large khớp 55.8% (CER 0.0906, WER 0.1482).
*Ghi chú phương pháp:* `data/phong_thi_data.csv` **không phải** tập câu đã sinh ra bộ mẫu
(đối chiếu tự động: 0/43 file khớp CER ≤ 0.20 — xem `offline_out/asr_ablation/align.json`), nên **không có
bảng CER so với bản chuẩn**. Chúng tôi dùng nhãn có sẵn + RTF + RAM và nói rõ giới hạn này trong báo cáo.

### Bảng 3 — Ảo giác: đầu vào KHÔNG phải tiếng người rõ (cổng BẬT vs TẮT)

| Chỉ số | Tắt cổng (cách làm phổ biến) | Bật cổng (hệ thống hiện tại) |
|:--|:--|:--|
| Số phép đo trên input nhiễu | 14 | 14 |
| Câu được sinh ra (bịa) | **2** | **0** |
| Cảnh báo sai (risk = Cheating/Suspicious) | **2** | **0** |
| File nói thật bị bỏ (chặn oan) | 0/12 | **0/12** |

Câu bịa thật đã ghi log: `"là tay bà mẹ đủ bề."` và `"là tay bà mẹ đủ đời."` — cả hai bị phân loại
**Cheating**, sinh ra từ đoạn 2–8 s của `data/video_test_demo.mp4` mà bên trong chỉ là nhiễu.
Chi tiết từng phép: `offline_out/asr_ablation/gate.json`.

### Bảng 4 — Độ bền theo SNR (8 câu nói thật trộn trên nền của chính clip demo)

Giao thức **self-consistency**: reference là văn bản model nhỏ trả lời trên file *sạch*;
"nguyên văn" = câu khi có nhiễu khớp hoàn toàn với câu khi sạch.

| SNR | Nền nhiễu phòng: qua cổng / nguyên văn / CER | Nền nhạc: qua cổng / nguyên văn / CER |
|:--|:--|:--|
| +6 dB | 7/8 · **6/8** · 0.148 | 8/8 · 3/8 · 0.208 |
| +3 dB | 7/8 · 4/8 · 0.221 | 8/8 · 1/8 · 0.275 |
| 0 dB | 6/8 · 3/8 · 0.355 | 8/8 · 1/8 · 0.513 |
| −3 dB | 2/8 · 0/8 · 0.800 | 5/8 · 0/8 · 0.712 |
| −6 dB | **0/8** · 0/8 · 1.000 | 3/8 · 0/8 · 0.884 |

**Đọc bảng — đây là điểm mạnh, không phải điểm yếu:** khi SNR thấp, chất lượng văn bản giảm đúng như dự
đoán, nhưng **cổng đóng lại và hệ thống im lặng** thay vì in một câu không ai nói vào biên bản.
Nền nhạc khó hơn nền nhiễu (Whisper bị nhạc dẫn sang văn phong "bản tin"), khớp với quan sát trên clip demo.

### Bảng 5 — End-to-end offline, 4 module + audio, trên 3 clip chuẩn của nhóm (M1, 29/10/2026)

| Clip | Frame | Thời gian xử lý | × thời lượng video | Kết luận actor-level |
|:--|:--|:--|:--|:--|
| exchange.mp4 (16.0 s) | 479 | 183.0 s | 11.4× | `student_02` → **suspicious_activity** (79 alert); `student_01` → c5 (60 alert) |
| cheatsheet.mp4 (15.0 s) | 448 | 171.4 s | 11.4× | cả hai actor → c5; có 1 sự kiện âm thanh |
| looking.mp4 (17.8 s) | 532 | 193.1 s | 10.9× | `student_01` 120 alert nhưng kết cục **c5** |

Trung bình một lượt suy luận (từ `timings_s` trong `report.json`): **vật cấm 30 ms · mặt 125–136 ms ·
hành vi 158–178 ms**. Mốc so sánh với bản cũ: 16.8 s video từng tốn 225.8 s → 104.1 s (2.17×).

> Khi đọc `alerts_by_module`: `face_verify` bằng đúng số frame vì pipeline ghi cả frame `status="info"`
> (trade-off đã ghi ở README §9 mục 11). Số để đếm cảnh báo thật là `report.json.actors`.

---

## 4. Trả lời trước những gì hội đồng có thể nghi ngờ

Xem `03_Phan_bien_QA.md` (22 câu, mỗi câu có số).

## 5. Hạn chế — nói trước, đừng để bị hỏi

1. **Điểm hành vi không tái lập giữa hai lần chạy cùng code** (đo 2026-10-28: c2 `0.6908` vs `0.6952`,
   suspicious `0.8584` vs `0.8149`). ⇒ Mọi so sánh A/B của chúng tôi chỉ dùng **quyết định/cảnh báo + thời gian**,
   không dùng điểm. Đây là tính liêm chính đo lường, không phải yếu điểm.
2. **macro-F1 locked test 0.767 < OOF 0.847**: chênh 7.96 điểm phần trăm giữa hai tập, cho thấy phân bố
   khó hơn khi giữ ngoài hoàn toàn. Chúng tôi báo cả hai số, không chọn số đẹp.
3. **Lỗi lớn nhất của classifier hành vi**: 19/67 mẫu `c5` bị gán thành siêu lớp nghi vấn (precision siêu lớp
   67.21% trong khi recall 87.23%) — hệ thống thiên về **cảnh báo để người giám thị xem lại**, không tự kết tội.
4. **ASR không cứu được bản thu xa**: với `data/video_test_demo.mp4` (giọng RMS 0.008–0.011 trên nền 0.0058,
   độ tự tin tiếng người 0.08–0.13), **cả ba cỡ model** đều trả câu bịa; large-v3 bịa kiểu
   `"Hãy subscribe cho kênh La La School…"`. Trong khi đó, với **nói thật thu gần mic** trộn trên đúng hai nền
   của clip này, hệ thống giữ nguyên văn **6/8 câu ở SNR +6 dB**; ở **−3 dB chỉ 2/8 còn qua cổng và 0/8 nguyên
   văn**, ở −6 dB (nền nhiễu) cổng đóng **0/8** — tức mất cảnh báo chứ không bịa cảnh báo (Bảng 4).
   ⇒ Nút thắt nằm ở **khâu thu âm**, không ở cỡ mô hình — đây là phát hiện thực
   nghiệm có số, không phải lời biện minh.
5. **Chưa có corpus audio kèm văn bản chuẩn** nên không báo cáo được CER/WER thật; các con số Bảng 2 chỉ so
   sánh *cùng điều kiện* giữa các cỡ model. Hướng sửa: ghi âm + gán nhãn phòng thi thật (đã ghi ở §6).
6. **Face mới chỉ là thầm dò nội bộ** (40% đầu / 60% sau trên cùng bản ghi, 17 actor–video) — chưa kiểm chứng
   chéo camera/phòng khác.
7. **Micro đi từ máy chạy ứng dụng**, không từ trình duyệt; truy cập từ máy khác sẽ không có âm thanh.

## 6. Hướng tiếp theo (nếu được hỏi "làm gì tiếp")

- **Front-end âm thanh**: beamforming/AEC khi dùng mảng mic, và đánh giá lại cổng ảo giác bằng CER trên một
  bộ mẫu **có** văn bản chuẩn (hiện chưa có — đã kiểm tra và loại `phong_thi_data.csv`).
- **Hành vi**: tăng dữ liệu cho lớp c3 (chỉ 8 mẫu locked test) và thử đối chứng bằng phép đo **tỉ lệ cảnh báo
  giả/giờ** thay vì macro-F1.
- **Triển khai**: đóng gói `requirements` theo 2 môi trường NumPy (Face cần 1.26.4) và chạy thử trên máy
  không có iGPU để đo ngưỡng tối thiểu phần cứng.

## 7. Phụ lục — lệnh tái lập từng con số

```bash
# Ablation ASR (Bảng 2, 3, 4)
.venv/bin/python scripts/asr_ablation.py --stage asr --model large
.venv/bin/python scripts/asr_ablation.py --stage asr --model medium
.venv/bin/python scripts/asr_ablation.py --stage asr --model small
.venv/bin/python scripts/asr_ablation.py --stage fusion
.venv/bin/python scripts/asr_ablation.py --stage robust
.venv/bin/python scripts/asr_ablation.py --stage gate
# -> offline_out/asr_ablation/*.json (mỗi file có timestamp + host + RAM đỉnh)

# Sức chứa realtime, không cần camera
.venv/bin/python scripts/replay_realtime.py data/exchange.mp4 --encode
.venv/bin/python -m backend.ai_services.pipeline.holistic_worker --selftest

# Bảng 5: offline trên 3 clip, có report.json theo actor
.venv/bin/python run_offline.py data/exchange.mp4   --output offline_out/bench_exchange
.venv/bin/python run_offline.py data/cheatsheet.mp4 --output offline_out/bench_cheatsheet
.venv/bin/python run_offline.py data/looking.mp4    --output offline_out/bench_looking

# Ablation thread (cần 5 lần chạy, đổi PROCTOR_CPU_THREADS)
PROCTOR_CPU_THREADS=1 .venv/bin/python scripts/bench_threads.py   # 145/68/76/60/118 ms @1080p
```
