# Ngân hàng phản biện — 26 câu hội đồng có thể hỏi, và cách trả lời ăn điểm

> Cách dùng: mỗi người trong team đọc to phần **Trả lời** cho tới khi nói được tự nhiên trong ≤ 30 giây.
> Quy tắc vàng: **(1)** không nhận sai, không chữa cháy; **(2)** luôn kèm số hoặc kèm "chưa đo";
> **(3)** nếu bị bắt lỗi thật → thừa nhận + nói đúng phạm vi của kết luận còn lại.
> Ký hiệu máy: **M1** = Intel Core Ultra 7 155H + iGPU Arc, 15 GB, Ubuntu/WSL2 (máy demo hôm nay).
> **M2** = AMD Ryzen 7 7435HS + RTX 4050, 11 GB thấy từ WSL2.

---

## A. Tính mới / vị trí so với các công trình khác

**A1. "Bài này khác gì so với các hệ proctoring thương mại (Mercu, ProctorU, HireVue) và các bài Eureka trước?"**
> Các hệ thương mại và hầu hết công trình đều tối ưu *độ chính xác của mô hình nhận diện hành vi*.
> Chúng tôi tối ưu *một ràng buộc mà chúng gần như không báo cáo*: **toàn bộ pipeline phải chạy trên thiết bị
> biên, trong ngân sách 50 ms/frame, với bốn mô hình cùng lúc**. đóng góp của chúng tôi là bốn kỹ thuật
> có số đo: cách ly GIL (p95 callback 83.0 → 24.0 ms trên M2), lịch nhịp nhân quả (offline 225.8 → 104.1 s),
> đặt mô hình đúng thiết bị + ngân sách thread (giấy 776 → 149 ms; 16 thread **chậm hơn** 8 thread), và cổng
> chặn ảo giác ASR. Ngoài ra chúng tôi **từ chối** dùng nhãn cấp video: mọi metric khoá theo `(video, actor_id)`.

**A2. "Vậy đóng góp khoa học ở đâu — các bạn chỉ đi 'tối ưu code'?"**
> Tối ưu ở đây là **có giả thuyết, có phép đo, có đường cong**. Ví dụ phát biểu "tăng số thread thì nhanh hơn"
> là sai với hệ của chúng tôi: 1→145 ms, 2→68 ms, 4→76 ms, 8→60 ms, 16→118 ms. Phát biểu "hình mờ do camera
> kém" cũng sai: nguyên nhân là vòng phản hồi REMB giữa độ bận CPU của server và quyết định hạ độ phân giải
> của trình duyệt. Cả hai đều là kết quả thực nghiệm tái lập được bằng lệnh trong phụ lục.

**A3. "Sao không dùng mô hình lớn hơn cho tốt hơn?"**
> Đã thử và **đo**, không phỏng đoán. Cùng 43 file có nhãn (22 gian lận / 21 bình thường), ba cỡ mô hình:
> PhoWhisper-small (OpenVINO iGPU), PhoWhisper-medium và whisper-large-v3 (CTranslate2 CPU int8) — Bảng 2.
>
> | Model | F1 | p50 s/câu | RTF | RAM đỉnh |
> |:--|:--|:--|:--|:--|
> | small (đang dùng) | **0.900** | **1.288** | **0.555** | **2094 MB** |
> | medium | 0.900 | 4.022 | 1.767 | 3106 MB |
> | large-v3 | **0.884** ↓ | 8.555 | 3.706 | 4212 MB |
>
> F1 **đi ngang rồi giảm** khi mô hình to lên; RTF tăng 6.6×, RAM tăng 2×. Lý do nằm ở dữ liệu: small và
> large-v3 chỉ khớp nhau 46.5% số câu, và cả ba đều sinh câu bịa trên bản thu mic xa — tức **nút thắt là khâu
> thu âm, không phải cỡ mô hình**. Đổi model to hơn chỉ mua thêm độ trễ mà không mua thêm độ đúng.
> Kết luận: với máy 15 GB không card rời, chỉ small + iGPU là đạt realtime.

---

## B. Phương pháp đo và metric

**B1. "Con số các bạn đo trên máy nào? Có áp dụng được cho máy khác không?"**
> Mỗi số đều gắn máy và ngày đo, đúng như footnote trên slide. M1 cho các số offline/thread/ASR; M2 cho các
> số trước–sau của cách ly GIL. Chúng tôi **không trộn** hai bảng. Điểm cần nói: kết luận *định tính* (GIL là
> nút thắt, nhiều thread hơn có thể tệ hơn) lặp lại trên cả hai máy, còn *định lượng* thì phụ thuộc phần cứng
> — đó là lý do chúng tôi để `PROCTOR_DEVICE`, `PROCTOR_CPU_THREADS` và có script đo lại trong 4 phút.

**B2. "Hai lần chạy cùng code cho điểm khác nhau — vậy số của bạn có tin được không?"**
> Đúng, điểm hành vi **không tái lập** (đo 2026-10-28: c2 0.6908 vs 0.6952, suspicious 0.8584 vs 0.8149).
> Vì vậy chúng tôi **cấm** việc A/B bằng điểm: mọi so sánh chỉ dùng **quyết định/cảnh báo + thời gian**.
> Đây là giới hạn chúng tôi tự phát hiện và ghi thẳng vào README, không phải bị hội đồng phát hiện.

**B3. "Vì sao macro-F1 locked test (0.767) thấp hơn OOF (0.847)? Các bạn chọn tập kiểm tra chưa đúng?"**
> Đó là hành vi bình thường của grouped split: locked test giữ ngoài *hoàn toàn* các nhóm người, nên phân bố
> khó hơn. Chúng tôi báo **cả hai** số và nói rõ chúng là **hai trò đánh giá khác nhau** (trang 22–24 bản cũ),
> không quy đổi thành "giảm tốc độ" hay "kết quả ablation".

**B4. "Precision của siêu lớp nghi vấn chỉ 67.21% — tức là 1/3 cảnh báo là sai?"**
> Vâng, và đó là **lỗi lớn nhất** chúng tôi đã định lượng: 19/67 mẫu `c5` bị gán siêu lớp nghi vấn.
> Thiết kế hệ thống là *cảnh báo sớm để người giám thị xem lại bằng chứng*, không phải tự kết luận gian lận
> (bản cũ đã ghi: "các kết quả hiện có chưa đủ để suy rộng thành kết luận tự động"). Trong kỳ thi thật,
> trade-off này nghiêng về recall vì bỏ sót đắt hơn kiểm tra thừa — nhưng chúng tôi không dùng lập luận đó
> để né con số 67.21%.

**B5. "Các bạn có so sánh với baseline không?"**
> Có, nhưng baseline là **chính phiên bản chưa tối ưu của chúng tôi** (same code path, đổi một biến):
> MediaPipe trong-tiến trình vs tiến trình riêng (`--no-isolate`), mỗi-frame vs lịch nhịp (225.8 vs 104.1 s),
> CPU vs iGPU cho detector giấy (1089 vs 319 ms), có vs không cổng ảo giác (Bảng 3). Chúng tôi **không** so
> với hệ thương mại vì không có quyền truy cập dữ liệu/phòng thi chuẩn hoá như nhau.

**B6. "Bao nhiêu dữ liệu là đủ để kết luận?"**
> Với hành vi: 75 video train / 28 video locked, 150 mẫu OOF và 56 mẫu locked theo actor — và chúng tôi nói
> rõ ở những lớp ít mẫu (c3 chỉ 8 mẫu locked ⇒ recall 62.5% có khoảng tin cậy rất rộng). Với hệ thống: mỗi
> phép đo lấy ≥10 lần sau warm-up và báo p50/p95/max. Với audio: 43 file có nhãn (22/21). Nói thẳng: **audio chưa có
> văn bản chuẩn để tính CER** — chúng tôi đã thử ánh xạ với `phong_thi_data.csv` và loại vì 0/43 file khớp.

---

## C. Mô hình và dữ liệu

**C1. "Sao không fine-tune lại YOLO/PhoWhisper cho dữ liệu phòng thi của mình?"**
> Đã fine-tune YOLO cho vật thể (checkpoint `best.pt`, mAP@0.5 0.93574 — metadata trong source, bản cũ trang 22).
> Với ASR, fine-tune cần phiên âm người thật; bộ mẫu hiện tại là câu kịch bản. Hướng đi đúng là **thu âm thật
> + gán nhãn**, và chúng tôi để trống phần đó thay vì báo số ảo.

**C2. "Dữ liệu do nhóm tự tạo — có tổng quát hoá được không?"**
> Bản cũ đã tự nhận: ảnh/điểm benchmark face là "đánh giá nội bộ thầm dò", "không chứng minh khả năng tổng
> quát sang camera, phiên quay hay quần thể mới". Chúng tôi giữ nguyên lập luận đó và đề xuất giao thức đánh
> giá chéo phòng (khác camera, khác ánh sáng, khác bàn) làm bước tiếp theo.

**C3. "Một tờ giấy mới có nhất thiết là gian lận không?"**
> Không. Đó là lý do có bước **cố định baseline (A)** và pipeline giấy so với mức chuẩn đầu phiên, đồng thời
> cảnh báo giấy chỉ là *chứng cứ vật thể*. Bản cũ ghi rõ: không có đủ ảnh/nhãn để tách lỗi "phát hiện giấy"
> khỏi lỗi "phân biệt giấy hợp lệ".

**C4. "Hai người trong khung hình thì gán hành vi cho ai?"**
> Cho actor, không cho video. Đó là toàn bộ lý do của tracking + ROI từng người + cửa sổ landmark 24 frame
> per actor, và của khoá `(video, actor_id)` trong `report.json`. Bản cũ nêu đúng tình huống: "hai người, chỉ
> một người quay sang không đồng nghĩa cả hai đều vi phạm".

---

## D. Hệ thống, hiệu năng, độ ổn định

**D1. "8 fps là các bạn tự đặt, hay có căn cứ?"**
> Căn cứ: `temporal_policy.json` của bundle ghi `target_fps = 8` — mô hình causal được train ở nhịp đó.
> Đẩy nhanh hơn làm **đổi điểm**, không chỉ đổi tốc độ. Và 8 fps là **nhịp lấy mẫu của tầng hành vi**, không
> phải fps của cả hệ — bản cũ đã nói rõ điều này, và nay chúng tôi có số đo end-to-end để chứng minh.

**D2. "Queue maxsize=2 drop-oldest — các bạn đang vứt dữ liệu mà gọi là tối ưu?"**
> Có chủ đích, và có đo: thà bỏ frame cũ nhất còn hơn trả lời quyết định bằng frame cũ (realtime có độ trễ).
> Nếu queue vô hạn, bộ nhớ phình và độ trễ tăng vô hạn khi một mô hình khựng. Số frame bị drop được **đếm và
> hiển thị công khai** trên hộp "🩺 Sức khoẻ module" — chúng tôi không giấu.

**D3. "Hệ thống crash/OOM chưa? Bằng chứng nào là chạy ổn định lâu?"**
> Đã gặp và đã sửa hai lỗi thật: (1) deadlock import `ultralytics` giữa hai thread ⇒ preload tuần tự trước khi
> sinh luồng; (2) OOM khi job offline chạy trong cùng tiến trình Streamlit (~3 GB model) ⇒ chuyển sang tiến
> trình riêng với `oom_score_adj=500` để con chết trước UI. Hiện CPU toàn app chỉ 2.5–3.2/22 core trên M1.
> Chưa đo: **stress test nhiều giờ liên tục** — đây là việc kế tiếp, tôi trả lời thật là chưa.

**D4. "MediaPipe chạy tiến trình riêng — nếu nó chết thì sao?"**
> Có watchdog: restart tối đa 3 lần, mỗi lần chỉ **một** tiến trình mới (có khoá, vì trước đây mỗi lần lỗi là
> sinh một con mới → `AttributeError` và fallback in-process giữ GIL 3–5 s). UI hiển thị `remote/tại chỗ/restart`
> để hội đồng thấy đúng lúc nó rơi. Selftest: `python -m backend.ai_services.pipeline.holistic_worker --selftest`.

**D5. "Sao không dùng STUN/TURN cho WebRTC?"**
> Trình duyệt và server cùng máy (WSL2), host candidate là đủ; thêm STUN chỉ tạo một vòng ICE ra Internet và
> **treo** nếu mạng công ty chặn UDP. Đây là quyết định có chủ đích, không phải thiếu.

---

## E. Âm thanh

**E1. "Micro của thí sinh ở phòng khác thì sao? WebRTC có chở âm thanh không?"**
> Không. Micro được thu **ở máy chạy ứng dụng** (PyAudio/sounddevice), còn hình đi qua WebRTC từ trình duyệt.
> Bản cũ đã cảnh báo: truy cập từ máy khác thì pipeline âm thanh không nhận tiếng của trình duyệt. Muốn giám
> sát tập trung thì phải đặt micro ở phòng thi, hoặc thêm kênh audio WebRTC (việc kế tiếp).

**E2. "F1 âm thanh 1.00 — các bạn tự tin quá không?"**
> Không, và chính bản cũ đã ghi là "tập nhỏ, câu ngắn, phân cách dễ". Chúng tôi chạy lại trên **43 file có
> nhãn** (22 gian lận / 21 bình thường) với **ba cỡ mô hình** (Bảng 2), và bổ sung hai phép đo bản cũ chưa có:
> **tỉ lệ sinh câu khi chỉ có nhiễu** (Bảng 3) và **độ bền theo SNR** (Bảng 4). Nói thẳng: F1 0.900 trên tập
> kịch bản tự tạo **không** chứng minh chất lượng trên giọng nói thật, tiếng lóng, hoặc bản ASR nhiều lỗi.

**E3. "Cổng chặn của các bạn có bỏ sót người nói nhỏ không?"**
> Câu hỏi đúng nhất để nghi ngờ, nên chúng tôi đo **cả hai chiều** và nói thẳng kết quả.
>
> - **Chiều không chặn oan:** 12 file nói thật đưa qua cổng → **12/12 vẫn đi qua** (`gate.json`,
>   `real_kept_with_gate=12/12`). Độ tự tin tiếng người của câu nói thật đo được 0.76–0.85, còn đoạn nhiễu
>   lọt lưới VAD chỉ 0.08–0.13 — hai nhóm **cách nhau rất xa**, nên ngưỡng 0.5 không nằm giữa cụm nào.
> - **Chiều có bỏ sót:** khi nói thật trộn dưới nền của chính clip demo, ở **SNR −6 dB cổng đóng toàn bộ
>   (0/8 qua cổng)** và ở −3 dB chỉ 2/8. Nghĩa là **nói quá nhỏ hoặc quá lẫn noise thì hệ thống im lặng** —
>   đó là giới hạn thật, chúng tôi không che.
> - **Vì sao im lặng vẫn là lựa chọn đúng:** phương án không cổng ở cùng vùng SNR thấp sinh ra câu bịa
>   ("là tay bà mẹ đủ bề." → phân loại **Cheating**) và in vào biên bản. Thà mất một cảnh báo còn hơn ghi một
>   tội danh không ai nói.
> - **Lever cho giám khảo:** `PROCTOR_ASR_MIN_SPEECH_PROB=0` tắt cổng để đối chiếu; `PROCTOR_VAD_THRESHOLD`
>   và auto-gain (`x1.95 / +5.8 dB` trên clip offline) xử lý nguyên nhân gốc là **mức thu âm thấp**.

**E4. "Thí sinh nói lảng/ nói tiếng Anh/ nói mã đề thì sao?"**
> Bộ từ khoá và PhoBERT chỉ phủ miền nội dung đã thu thập. Bản cũ ghi rõ: "không có locked test câu văn độc
> lập" và kết luận âm thanh chỉ nên dùng ở mức **tham khảo**. Vì vậy cảnh báo âm thanh không đủ để gán c1–c5
> cho một actor cụ thể trong cảnh nhiều người.

---

## F. Đạo đức, quyền riêng tư, triển khai thật

**F1. "Hệ thống giám sát người thi — quyền riêng tư và giả định vô tội xử lý thế nào?"**
> Ba lớp: (1) **không tự kết tội** — output là cảnh báo kèm bằng chứng (khung hình, dòng log, actor id) để con
> người quyết định; (2) ảnh đăng ký khuôn mặt và log phiên nằm trong `data/`, **không vào Git**, và UI cho phép
> **xoá** từng học sinh có xác nhận; (3) mọi kết luận thống kê đều kèm giới hạn (mục Hạn chế, slide 9).
> Nếu triển khai thật, bắt buộc thêm: sự đồng ý bằng văn bản, thời gian lưu có hạn, và quy trình phúc khảo.

**F2. "Nếu camera nhìn không thấy mặt (cúi thấp, quay đi) thì hệ thống nói gì?"**
> Không suy diễn. Track mất thì `is_present=False`, quá 0.75 s thì khung xương bị xoá thay vì đóng băng ở vị trí
> cũ; không có landmark thì không có quyết định hành vi. Đây là hành vi trung thực nhất: "không thấy" ≠ "không
> gian lận", và cũng ≠ "có gian lận".

**F3. "Thí sinh dùng điện thoại nhưng để ngoài tầm camera thì sao?"**
> Ngoài phạm vi cảm biến — hệ thống chỉ kết luận trong vùng quan sát. Nói rõ như vậy tốt hơn là hứa.

---

## G. Câu hỏi "bẫy" và cách xử lý

**G1. "Số liệu này các bạn tự đo hay chép của nhau?"**
> Mỗi con số trong phụ lục đều kèm **lệnh tái lập**. Mời hội đồng chọn một số bất kỳ, chúng tôi chạy lại ngay
> trên máy này trong vòng dưới 5 phút (`replay_realtime.py`, `holistic_worker --selftest`, `bench_threads.py`,
> `asr_ablation.py`).

**G2. "Tại sao slide không có kết quả so với công trình A, B, C?"**
> Vì không có bộ dữ liệu và điều kiện phần cứng chuẩn hoá chung. Chúng tôi so với **chính phiên bản chưa tối
> ưu** của mình, đó là ablation trung thực nhất có thể trong phạm vi đề tài sinh viên.

**G3. "Demo hôm nay là thật hay video dựng sẵn?"**
> Cả hai, và nói rõ cái nào là cái nào: 90 giây demo gồm **ứng dụng chạy thật** với camera + telemetry trực
> tiếp (0:00–0:45), rồi **20 giây video kết quả của luồng offline** (0:45–1:10, burn overlay + cột cảnh báo +
> audio để nguyên bản), cuối cùng là `report.json` mở trên màn hình. Chúng tôi nói đang ở chế độ nào, và
> không trình chiếu video mà gọi là realtime.

**G4. "Các bạn nói 'không cloud' — vậy nếu trường muốn lưu bằng chứng để phúc khảo?"**
> Ngay bây giờ đã có: `events.jsonl`, `report.json` theo actor, `annotated.mp4`. Việc đó chạy trên máy chủ
> trong trường, không cần cloud. Đây là lợi thế của thiết kế biên: bằng chứng nằm trong quyền kiểm soát của
> trường và đúng quy định bảo vệ dữ liệu.

**G5. "Giả sử hội đồng thấy phần mềm chạy mượt — biết đâu các bạn tắt bớt module?"**
> UI có multiselect **"Module bật"** và hộp **🩺 Sức khoẻ module** hiển thị fps/độ trễ/số frame rớt **của từng
> luồng**, cộng thêm dòng `remote/tại chỗ/restart` của Holistic. Trạng thái bật/tắt nằm ngay trên màn hình,
> và `report.json` ghi lại `config` (modules, strides, device, model path) của đúng lần chạy.

**G6. "Điểm yếu lớn nhất của đề tài là gì?"**
> Ba điểm, theo thứ tự nghiêm trọng: (1) dữ liệu hành vi do nhóm tự tạo, chưa kiểm chứng chéo phòng;
> (2) precision cảnh báo 67.21% — còn nhiều cảnh báo sai; (3) âm thanh chưa có corpus có văn bản chuẩn và
> chưa tách được **ai** nói. Chúng tôi chọn cách trình bày giới hạn trước khi bị hỏi.

---

## H. Checklist 60 giây trước khi lên sân khấu

- [ ] Đã tắt mọi tiến trình Streamlit khác (`pgrep -af "[s]treamlit"`) — hai instance làm hỏng micro.
- [ ] Mở sẵn app, **chưa** bấm START; mở sẵn `playback_h264.mp4` của clip 20 s làm phương án B.
- [ ] Mic: panel hiển thị `RMS` và ngưỡng 0.02; nếu phòng ồn, đặt `PROCTOR_VAD_THRESHOLD=0.012`.
- [ ] Đăng ký trước 2 khuôn mặt để nhánh face có nội dung demo.
- [ ] Bảng 2/3/4/5 đã điền số thật, footnote ghi máy + ngày + lệnh.
- [ ] **Bấm giờ thật, chạy đủ 2 lần**: mốc cứng là 2:50 phải rời slide sang demo, 4:20 quay lại trang 7.
      Trễ mốc 2:50 = cắt trang 6, không cắt trang 7/8.
- [ ] Người nói chính + người demo là **hai người khác nhau** (5 phút rất chặt).
- [ ] Đã thuộc ba câu "không được nói quá": không nói "100% ở SNR thấp", không nói "nhanh 12 lần", không nói
      "điểm tái lập được" (xem `README.md` cùng thư mục).
