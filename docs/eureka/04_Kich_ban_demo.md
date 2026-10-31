# Kịch bản DEMO 90 giây (nằm trong phần 5 phút)

> **Người bấm máy và người nói là hai người khác nhau.** Người nói không quay lại bàn phím.
> Mọi thứ phải mở sẵn từ trước; demo chỉ *bấm*, không *tìm*.

## 0. Chuẩn bị (trước 15 phút)

```bash
# 1) Không để hai instance cùng chạy (hai Streamlit làm hỏng micro)
pgrep -af "[s]treamlit" || echo "sạch"

# 2) Môi trường cho phòng thi thật (ồn hơn phòng demo)
export PROCTOR_VAD_THRESHOLD=0.012      # nếu nền phòng ồn
export PROCTOR_REMB_FLOOR=1200000       # giữ 960x540, chống Chrome tự hạ độ phân giải

# 3) App đã mở, CHƯA bấm START
.venv/bin/streamlit run app_webrtc.py

# 4) Phương án B: video 20s của luồng offline (đã có sẵn playback_h264.mp4)
#    và một bản report.json của lần chạy thành công
```

Checklist nhanh: ☐ 2 học sinh đã đăng ký khuôn mặt ☐ camera hoạt động ☐ panel "🩺 Sức khoẻ module"
đang hiển thị ☐ `playback_h264.mp4` mở sẵn ở tab khác ☐ tắt thông báo (Do Not Disturb).

---

## 1. 0:00–0:15 — Ba chức năng mới trên giao diện

Chỉ vào (theo thứ tự):
1. **⚙️ Module realtime → "Module bật"**: "Hội đồng có thể tắt từng mô-đun ngay khi đang chạy — đây là cách
   chúng tôi đo ablation, và cũng là cách chứng minh hệ thống *đang thật sự chạy mô hình*."
2. **👥 Học sinh đã đăng ký**: "Quản lý danh tính ngay trong UI: xem ảnh, xoá có xác nhận, nạp lại FAISS
   không cần khởi động lại." (Nhấn **♻️** để nạp lại — panel cập nhật số khuôn mặt.)
3. **🩺 Sức khoẻ module**: "fps, độ trễ p50/p95 và **số frame bị vứt** của từng luồng. Chúng tôi công khai
   phần mình bỏ, thay vì chỉ khoe phần chạy được."

## 2. 0:15–0:45 — Luồng realtime

Bấm **START**. Nói trong khi hình lên:

- "Bốn mô-đun đang chạy đồng thời trên chính chiếc laptop này, không có card rời, không gọi cloud."
- Chỉ vào **📥 WebRTC vào: 960×540 @ ~20 fps** → "đây là bằng chứng cái trình duyệt đang gửi lên, thứ mà
  trước đây tụt xuống 640×360 và làm mờ hình."
- Chỉ khung xương + `Track 1 | student_01 | 0.86` → "mỗi người một danh tính; quyết định là **theo người**,
  không theo video."
- Nếu hội đồng hỏi "sao không thấy cảnh báo": "clip này hai bạn đều được kết luận `c5` — không có cảnh báo
  là kết quả đúng, không phải hệ thống chết. Hội đồng có thể thấy `alerts=...` trong `report.json`."

> **Mẹo ăn điểm:** mở terminal, chỉ dòng `[WEBRTC] Sàn REMB 1200 kbps` và (nếu có)
> `[WEBRTC] đổi kích thước vào: ...` — "đây là log chúng tôi thêm để *bắt quả tang* lỗi giao thức".

## 3. 0:45–1:10 — 20 giây video của luồng offline

Chuyển tab sang `playback_h264.mp4` (H.264 để trình duyệt/Windows phát được). Nói:

- "Cùng một pipeline, chạy trên file. Overlay, bảng cảnh báo và **âm thanh gốc** được burn/ghép vào video,
  nên biên bản có thể phát lại được."
- Dừng ở khung có `🚨 GIAN LẬN HÀNH VI: student_02: EXCHANGE` → "bảng log này chỉ làm mới 1 lần/giây video
  và gộp dòng trùng — chi tiết nhỏ nhưng quyết định việc người giám thị **đọc kịp**."

## 4. 1:10–1:30 — Bằng chứng dạng số

Mở `report.json` của lần chạy (hoặc `events.jsonl`), chỉ vào:

```
"actors": { "student_01": {"final_class": "...", "evidence_score": ..., "first_flag_frame": ...},
            "student_02": {...} }
"alerts_by_actor": {...}
"config": {"modules": ..., "strides": ..., "device": ...}
```

- "Mọi kết luận khoá theo `(video, actor_id)`, kể cả người bị kết `c5`. Và `config` được ghi lại để lần sau
  chạy đúng điều kiện — đó là toàn bộ tiêu chí tái lập của tụi em."

---

## 5. Phương án dự phòng (bắt buộc phải có)

| Sự cố | Xử lý ngay | Câu nói |
|:--|:--|:--|
| Camera không lên / bị từ chối quyền | Chuyển sang `playback_h264.mp4` + ảnh chụp panel telemetry đã chụp trước | "Phòng thi thật có camera, nhưng nếu hội đồng muốn thấy đúng pipeline thì đây là kết quả burn từ cùng code path." |
| Micro không thu (WSL2 thiếu plugin ALSA) | Tắt module `audio` bằng multiselect, demo 3 module hình | "Micro thu ở máy chạy ứng dụng chứ không đi qua trình duyệt — tụi em ghi rõ giới hạn này trong báo cáo." |
| App đứng / hết RAM | Không khởi động lại trên sân khấu. Mở video + report đã lưu | "Đây là lý do job offline chạy ở tiến trình riêng với `oom_score_adj`, để tiến trình phụ chết trước UI." |
| Hội đồng yêu cầu chạy lại một con số | Chạy `holistic_worker --selftest` (20 s) hoặc `replay_realtime.py --fast` (60–90 s) | Nói rõ máy đang demo là M1 (iGPU) và số này thuộc bảng nào. |

## 6. Những gì **không** được làm trong demo

- Không hứa "mọi kỳ thi đều phát hiện đúng". Không dùng từ "chấm dứt gian lận".
- Không nói realtime fps khi đang chiếu video dựng sẵn — nói rõ đang ở chế độ nào (G3 trong ngân hàng phản biện).
- Không bật nhạc/loa to khi mic đang mở — pipeline audio sẽ thu lại và sinh cảnh báo ngay trên sân khấu.
