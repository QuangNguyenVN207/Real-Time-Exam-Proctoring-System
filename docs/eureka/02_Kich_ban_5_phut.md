# Kịch bản trình bày 5 phút (10 slide + demo 90 giây)

> Nguyên tắc theo đúng đề bài: **không giới thiệu lại dự án** (hội đồng đã đọc bản PDF). Vào thẳng
> "chúng tôi đã đóng ba khoảng trống mà chính bản cũ để ngỏ".
> **Quỹ thời gian: 5:00 tổng = ~2:50 slide + 90 giây demo + 40 giây số liệu & chốt.** Đánh số dưới đây = đúng
> số trang trong `Eureka_delta_slides.pptx`, không lệch nhau.

| Trang pptx | Thời điểm | Nội dung | Câu chốt phải nói |
|:--|:--|:--|:--|
| 1 | 0:00–0:15 | Bìa: "Bốn mô hình AI, một chiếc laptop" + 50 ms và 2.5–3.2/22 core | "Em không lặp lại kiến trúc. Bản cũ tự khai ba chỗ chưa đo — tụi em trả lời cả ba bằng số." |
| 2 | 0:15–0:50 | Ba khoảng trống bản cũ tự khai (§5.9, §5.7, §4.9) + bằng chứng mới | "Ba câu đó do chính tụi em viết, nên tụi em có trách nhiệm đo lại." |
| 3 | 0:50–1:25 | Cách ly GIL: trong tiến trình vs tiến trình riêng | "Cùng model, cùng code suy luận — chỉ khác cách tổ chức tiến trình — p95 giảm 3.5 lần." |
| 4 | 1:25–1:50 | Lịch nhịp nhân quả: 225.8 s → 104.1 s | "8 fps không phải để tiết kiệm, mà vì mô hình được train ở 8 fps." |
| 5 | 1:50–2:10 | Ablation thread: 145/68/76/60/118 ms | "Đường cong không đơn điệu: 16 thread **chậm hơn** 8 thread." |
| 6 | 2:10–2:30 | Đổi chỗ mô hình: 5.2× · 3.4× · 8.1× | "Thiết bị và số thread là biến số thực nghiệm, không phải tham số triển khai." |
| 7 | 2:30–2:50 | ASR 3 cỡ model — nói MỘT CÂU rồi dừng, để bảng số tới sau demo | "To hơn không mua được độ đúng — tụi em có số, xin nói ngay sau hai phút demo." |
| — | **2:50–4:20** | **DEMO 90 giây** (rời slide, xem `04_Kich_ban_demo.md`) | — |
| 7 | 4:20–4:40 | Quay lại trang 7: F1 0.900 / 0.900 / 0.884, RTF 0.56 / 1.77 / 3.71, RAM 2.09 / 3.11 / 4.21 GB | "large-v3 F1 THẤP HƠN và RTF 3.7 — máy này không chạy realtime được." |
| 8 | 4:40–4:50 | Cổng chặn ảo giác (2 → 0, 12/12 giữ) + "REMB giả nghẽn" | "Whisper luôn trả văn bản. Hệ thống đáng tin là hệ thống biết im lặng." |
| 9 | 4:50–4:55 | End-to-end: 3 clip, 4 module, actor-level | "Toàn bộ số này đo trên một chiếc laptop không card rời, không gọi cloud." |
| 10 | 4:55–5:00 | Giới hạn + câu kết | "Chúng tôi không thắng bằng mô hình to hơn." |

> **Nếu ban tổ chức cắt còn 4 phút:** bỏ trang 6 (dồn vào trang 5), rút demo còn 60 giây (chỉ realtime, bỏ
> 20 giây video offline), trang 7 chỉ nói "F1 không tăng khi model to hơn".
> **Nếu được thêm giờ:** trang 7 và 8 là hai trang nói sâu nhất (phương pháp + số), đừng cắt chúng.

---

## Trang 1 — Bìa (15 giây)

```
BỐN MÔ HÌNH AI, MỘT CHIẾC LAPTOP
50 ms               ngân sách cho MỖI khung hình khi 4 mô hình cùng chạy
2.5–3.2 / 22 core   tổng CPU của cả ứng dụng (đo trên máy này)
```

**Nói:** "Dạ em chào hội đồng. Em không trình bày lại kiến trúc — hội đồng đã đọc bản báo cáo. Trong hai tuần
qua tụi em đi đóng ba khoảng trống mà **chính bản cũ tự khai là chưa đo**, và mang được bằng chứng tới hôm nay."

---

## Trang 2 — Ba khoảng trống, một đêm đo

```
BẢN CŨ TỰ KHAI (trang 20, 26, 27):
  §5.9  "8 FPS là cấu hình lấy mẫu tầng pose, không phải FPS đã đo cho cả bốn mô-đun đồng thời"
  §5.7  "Thời gian CPU trung bình 15.69 giây/cặp, chưa đáp ứng xử lý tuần tự thời gian thực"
  §4.9  "Việc có một lệnh khởi động thành công chưa đủ để xác nhận đang chạy mô hình thực"

BẢN NÀY:  đo lại end-to-end trên 2 máy, có harness tái lập được
```

**Nói:** "Ba câu này là chữ của tụi em, nên tụi em không chữa cháy — tụi em đo. Cả ba đều có số ở phần sau."

---

## Trang 3 — Cách ly GIL: đa luồng Python không phải là song song

```
VẤN ĐỀ           MediaPipe Tasks giữ GIL suốt lượt suy luận: 100–113 ms/tick (cold 886 ms)
                 ⇒ callback WebRTC bị chặn ⇒ streamlit_webrtc phát lại _last_out_frame (frame CŨ)
                 ⇒ hình vừa giật vừa trễ dần

GIẢI PHÁP        Holistic ra tiến trình con (socket AF_UNIX + pickle, nice 10, 4 core,
                 tự dựng lại ≤3 lần, fallback in-process)

KẾT QUẢ (M2 / RTX 4050)          GIL giữ: 100–113 ms → 0
                                 callback p50: 32.9 → 13.3 ms  |  p95: 83.0 → 24.0 ms
                                 overlay p95: 41.5 → 6.0 ms
KẾT QUẢ (M1 / iGPU, máy demo)    ~31 ms/tick, remote=3 fallback=0
```

**Nói:** "Đây là lỗi của **kiến trúc ngôn ngữ**, không phải lỗi mô hình. Hai hệ dùng cùng model, cùng độ chính
xác, nhưng một hệ giật — vì GIL. Chúng tôi đo được cả độ dài thời gian GIL bị giữ, và chạy
`holistic_worker --selftest` ngay trước mặt hội đồng được."

---

## Trang 4 — Lịch nhịp nhân quả

```
Ràng buộc 1 (khoa học): causal model train ở target_fps = 8  → đẩy 30 fps = sai phân bố
Ràng buộc 2 (luật chơi): frame tại t+1 chưa tồn tại          → mask tương lai
Ràng buộc 3 (hệ thống):  ngân sách 50 ms/frame @ 20 fps      → queue maxsize=2, rơi frame CŨ NHẤT

Lịch: hành vi 8 fps · thị giác 10 fps · giấy mỗi 5 frame · callback p50 4.5 ms

Offline 16.8 s video, 4 module:  225.8 s  →  104.1 s   (2.17×)
```

**Nói:** "Chi tiết quan trọng nhất: chúng tôi **không** tăng fps cho mượt. Tăng stride của tầng hành vi làm
**đổi điểm** c2/c3, vì cửa sổ 24 landmark frame bị co giãn. Nên 8 fps là con số trung thực với lúc train,
không phải con số ăn may."

---

## Trang 5 — Đo trước khi chia tài nguyên

```
Mặc định của PyTorch/ONNX Runtime = chiếm hết 22 thread → 4 mô-đun giành nhau

Ablation thread, YOLOv8n @1920×1080 (máy này, scripts/bench_threads.py):
   1 thread  145 ms
   2 thread   68 ms
   4 thread   76 ms
   8 thread   60 ms
  16 thread  118 ms   ← ĐƯỜNG CONG KHÔNG ĐƠN ĐIỆU
```

**Nói:** "Nếu hội đồng hỏi 'sao không để mặc định': mặc định chiếm hết 22 thread, và 16 thread **chậm hơn**
8 thread, gần gấp đôi cấu hình chúng tôi chọn. Đây là phát hiện thực nghiệm, tái lập bằng đúng một lệnh."

---

## Trang 6 — Đổi chỗ mô hình sang đúng thiết bị

```
Detector giấy   CPU → iGPU OpenVINO            776 → 149 ms   (5.2×)  đo TRONG ứng dụng
Cùng phép đổi   harness replay (không camera)  1089 → 319 ms  (3.4×)  để đối chiếu
FaceNet         CPU → CUDA (M2)                232 → 28.7 ms  (8.1×)

select_device().kind chọn model + thiết bị theo PHẦN CỨNG (cuda / intel / cpu), không theo OS
```

**Nói:** "Cùng một mô hình, đặt sai thiết bị thì sai luôn kết luận. Chúng tôi chọn thiết bị bằng số đo, không
bằng mặc định của thư viện."

---

## Trang 7 — ASR: to hơn không có nghĩa là tốt hơn

```
Cùng 43 file có nhãn (22 gian lận / 21 bình thường), cùng bộ từ khoá + PhoBERT, chỉ đổi model:

                     F1      P       R      p50 s/câu   RTF    RAM đỉnh
PhoWhisper-small    0.900  1.000   0.818     1.29      0.56    2.09 GB   ← iGPU, ĐANG DÙNG
PhoWhisper-medium   0.900  1.000   0.818     4.02      1.77    3.11 GB   ← CPU int8
whisper-large-v3    0.884  0.905   0.864     8.56      3.71    4.21 GB   ← CPU int8

F1 đi ngang RỒI GIẢM khi model to lên; RTF tăng 6.6 lần, RAM tăng 2 lần.
Tham chiếu chéo: small / medium chỉ khớp large-v3 ở 46.5% / 55.8% số câu (CER 0.09).
KHÔNG có bảng CER so với văn bản chuẩn: phong_thi_data.csv không phải câu đã sinh ra bộ mẫu (0/43 khớp).
```

**Nói (lần 1, lúc 2:30 — chỉ một câu):** "Chúng tôi có thử model to hơn, và số nói rằng nó không mua được độ
đúng. Em xin để bảng này lại, nói ngay sau hai phút demo."
**Nói (lần 2, lúc 4:20):** "F1 đi ngang rồi **giảm**: large-v3 0.884 so với small 0.900, còn RTF lên 3.7 — máy
này không chạy realtime được. Nút thắt nằm ở **khâu thu âm**, không ở cỡ mô hình: trên bản thu mic xa, cả ba
cỡ đều bịa. Nên chúng tôi đầu tư vào cổng chặn và chuẩn hoá âm lượng, không vào thông số model."

---

## Trang 8 — Cổng chặn ảo giác + "REMB giả nghẽn"

```
WHISPER LUÔN TRẢ VĂN BẢN — kể cả khi đầu vào chỉ có nhiễu
   đoạn nhiễu 0.75 s → "là tay bà mẹ đủ bề." → bị phân loại Cheating (cảnh báo SAI do hệ thống)

3 lớp sửa:
   (1) chuẩn hoá mức RMS cả file video (--audio-gain; đo x1.95 / +5.8 dB trên clip offline)
   (2) chuẩn hoá mức cục bộ đoạn tiếng trước khi vào ASR
   (3) CỔNG: đoạn tiếng ≥0.8 s VÀ độ tự tin tiếng người (Silero) TB ≥0.5

KẾT QUẢ  nhiễu (14 phép):  câu bịa 2 → 0  |  cảnh báo sai 2 → 0
         nói thật (12 phép): 12/12 VẪN QUA CỔNG — không chặn oan
         SNR +6 dB: 6/8 nguyên văn (CER 0.15) | SNR −3 dB: 2/8 qua cổng, 0/8 nguyên văn

REMB GIẢ NGHẼN  Python bận → gói RTP đến dồn cục → aiortc báo nghẽn GIẢ → Chrome
                 scaleResolutionDownBy 960×540 → 320×180 ⇒ hình MỜ, không phải do camera
SỬA  sàn REMB 1.2 Mbps (PROCTOR_REMB_FLOOR) + MIN_BITRATE VP8 250 → 1200 kbps
KIỂM CHỨNG  log "đổi kích thước vào" + panel "WebRTC vào: 960×540 @ 20 fps"
```

**Nói:** "Một hệ thống giám sát mà sinh ra câu nói không ai nói thì còn tệ hơn hệ thống chậm. Cổng này **không
chặn oan tiếng người** — 12/12 file nói thật vẫn đi qua; nó chặn đoạn không phải tiếng người, và khi tiếng
người lẫn quá sâu thì hệ thống **chọn im lặng** thay vì bịa. Vế bên phải là một lỗi chỉ tồn tại ở hệ realtime:
CPU của chính chúng tôi làm trình duyệt tin rằng mạng nghẽn — pipeline offline không có hiện tượng này."

---

## DEMO — 90 giây (2:50 → 4:20)

Chuyển thẳng sang ứng dụng đang mở sẵn. Kịch bản chi tiết, env cần đặt trước và bảng phương án dự phòng:
`04_Kich_ban_demo.md`. Trong lúc 20 giây video offline chạy, quay lại **trang 7** để nói số ASR.

---

## Trang 9 — End-to-end, bốn module, một chiếc laptop

Bảng 5 (ba clip chuẩn, `run_offline.py`, 4 module + audio) + ảnh chụp hộp **Sức khoẻ module** trên UI
(fps/độ trễ/rớt frame từng luồng, running).

```
exchange    479 frame · 183.0 s · student_02 → suspicious_activity (79 alert)
cheatsheet  448 frame · 171.4 s · cả hai actor → c5
looking     532 frame · 193.1 s · student_01 120 alert nhưng kết cục c5
vật cấm 30 ms · mặt 125–136 ms · hành vi 158–178 ms

máy tham chiếu (M1): Intel Core Ultra 7 155H (22 logical CPU) + iGPU Arc, 15 GB RAM, Ubuntu/WSL2
mọi kết luận khoá theo (video, actor_id) — kể cả actor bị kết c5
```

**Nói:** "Toàn bộ số liệu trên màn hình này đo trên đúng một chiếc laptop không có card rời, không gọi cloud,
và truy vết được từng thí sinh chứ không chấm điểm cho cả video."

---

## Trang 10 — Hạn chế chúng tôi nói trước + câu kết

```
• Điểm hành vi KHÔNG tái lập giữa 2 lần chạy cùng code (c2 0.6908 vs 0.6952)
  ⇒ chỉ A/B bằng quyết định/cảnh báo + thời gian
• locked test macro-F1 0.767 < OOF 0.847  ⇒ báo cả hai số, không chọn số đẹp
• Lỗi lớn nhất: 19/67 mẫu c5 bị gán siêu lớp nghi vấn (P 67.2% / R 87.2%)
  ⇒ thiết kế là "cảnh báo để người giám thị xem lại", không tự kết tội
• ASR không cứu được bản thu xa: cả 3 cỡ model đều bịa; nói thật thu gần mic giữ nguyên văn 6/8 ở
  SNR +6 dB, và hệ thống CHỌN IM LẶNG ở SNR thấp ⇒ nút thắt là KHÂU THU ÂM
• Không có bảng CER so với văn bản chuẩn (đã kiểm tra và loại phong_thi_data.csv, 0/43 khớp)
• Face mới là thầm dò nội bộ (17 actor–video, 40/60 split)

KẾT: Chúng tôi không thắng bằng mô hình to hơn.
     Chúng tôi thắng bằng cách tổ chức để BỐN mô hình chạy thật trên thiết bị biên —
     và chứng minh từng phát biểu bằng một phép đo tái lập được.
```

---

## Ghi chú dựng slide (`Eureka_delta_slides.pptx`)

- 10 trang, 1280×720, dựng đúng theo thứ tự trên; mỗi trang đều ghi máy + ngày ở footnote.
- Trang 7 và 8 dày số nên thân chữ 12–15 pt, tiêu đề 26–28 pt — ép 24 pt cho toàn thân sẽ tràn khung.
- Không dùng ảnh chụp màn hình resize; nếu cần ảnh thì crop đúng vùng (panel telemetry, khung xương) và ghi rõ
  "hình chụp ngày … trên máy …".
- Khi nói, **đừng đọc slide**: slide đã có số, miệng chỉ nói nguyên nhân và kết luận.
