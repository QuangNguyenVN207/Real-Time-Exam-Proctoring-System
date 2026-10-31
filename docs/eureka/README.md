# Tài liệu chuẩn bị Euréka — vòng chung kết (5 phút + phản biện)

Bốn file dưới đây là bộ hồ sơ làm việc của team. Mọi con số đều đo trên máy thật và có file JSON kèm theo.

| File | Dùng để | Đọc trước |
|:--|:--|:--|
| `01_Bao_cao_Eureka_v2.md` | Bản report mới: 6 đóng góp có số, bảng delta so với bản PDF hội đồng đã đọc, Bảng 2/3/4/5 | cả team |
| `02_Kich_ban_5_phut.md` | Kịch bản nói từng slide + phân bổ thời gian (kèm phương án cắt về 5 phút) | người nói chính |
| `03_Phan_bien_QA.md` | 26 câu hội đồng có thể hỏi, mỗi câu có trả lời ≤ 30 giây kèm số hoặc chữ "chưa đo" | cả team, luyện nói |
| `04_Kich_ban_demo.md` | Kịch bản demo 90 giây, env cần đặt trước, bảng phương án dự phòng | người demo |
| `Eureka_delta_slides.pptx` | 10 slide trình chiếu (1280×720) | — |

## Số liệu nằm ở đâu (bằng chứng gốc)

- ASR 3 cỡ mô hình + đối chiếu: `offline_out/asr_ablation/{asr_small,asr_medium,asr_large,fusion}.json`
- Cổng chặn ảo giác (14 phép nhiễu + 12 phép nói thật): `offline_out/asr_ablation/gate.json`
- Độ bền theo SNR (self-consistency): `offline_out/asr_ablation/robust.json`
- Lý do không có bảng CER so với bản chuẩn: `offline_out/asr_ablation/align.json` (0/43 file khớp)
- End-to-end 3 clip: `offline_out/bench_{exchange,cheatsheet,looking}/report.json`
- Lệnh tái lập: `scripts/asr_ablation.py --stage {asr|fusion|robust|gate}`, `scripts/bench_threads.py`,
  `run_offline.py <video>`, `python -m backend.ai_services.pipeline.holistic_worker --selftest`

## Máy đã đo

- **M1** = Intel Core Ultra 7 155H + iGPU Arc, 22 logical CPU, 15 GB RAM, Ubuntu/WSL2 — máy demo, nguồn của
  các số offline / thread / ASR.
- **M2** = AMD Ryzen 7 7435HS + RTX 4050 — nguồn của các số trước–sau cách ly GIL.
  Hai bảng **không trộn lẫn**; mỗi slide đều ghi máy + ngày.

## Ba điều không được nói quá

1. Không khẳng định "nhận đúng 100% ở SNR thấp": đo thật là −3 dB chỉ 2/8 đoạn qua cổng, 0/8 nguyên văn.
2. Không đổi "15.69 s/cặp (CPU, bản cũ)" thành "nhanh 12 lần": khác đơn vị và khác máy, chỉ nói *hết chặn realtime*.
3. Không gọi điểm hành vi là tái lập được; A/B chỉ dùng quyết định/cảnh báo + thời gian.
