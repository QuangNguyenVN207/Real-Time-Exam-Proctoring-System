# XGBoost C3 live-camera recovery plan

## 1. Mục tiêu và phạm vi

Khôi phục XGBoost hiện tại để nhận C3 trên camera thật với đúng hai actor. Làm theo thứ tự:

1. **Stage A — Observe:** ghi đủ dữ liệu và tạo replay đỏ từ camera fail.
2. **Stage B — Repair:** sửa C3/C2 relation, gate, baseline và timebase; replay phải xanh.
3. **Stage C — Retrain only if needed:** chỉ retrain XGBoost khi runtime đã đúng nhưng score vẫn không tách được C3/C5.
4. **Stage D — Camera gate:** kiểm tra camera liên tục trước khi promotion.

Ngoài phạm vi: ST-GCN++, attention, YOLO C1/C4, full-dataset audit, đổi manifest, benchmark locked test trước khi Stage B xanh.

Không hạ threshold để tạo flag. Mọi threshold thay đổi phải đến từ calibration data và cùng replay policy deployment.

## 2. Contract đã khóa

- C3 là directed relation: `A nhìn/quay về B` chỉ gán C3 cho A.
- Chỉ có tối đa hai actor. Khi cả hai observable, tạo hai edge `A:B` và `B:A`.
- Peer mất quá thời gian cho phép: output `unknown`; không tự gán C5.
- Track ID đổi: xóa state relation cũ và calibration lại.
- Demo có neutral calibration 2 giây trước khi READY.
- C3 dùng head/body feature. Hand không hard-veto C3.
- C2 dùng hand/pair feature. Khi C2 đủ evidence, C2 thắng C3 cho cả pair.
- Thứ tự resolver: `C2 > C3 > suspicious_activity > C5`; `unknown` đứng ngoài class resolver.
- Camera khoảng 10 FPS: duration, staleness và latency tính bằng timestamp milliseconds, không tính số frame.

## 3. Ký hiệu và tham số

### 3.1 Đơn vị

- `t`: timestamp hiện tại, milliseconds.
- `A`, `B`: actor ID ổn định từ tracker.
- `e_AB`: directed edge “A hướng về B”.
- `O_A(t)`: A observable tại `t`; pose head, shoulders, hips đủ reliability.
- `O_AB(t)`: A và B observable, hoặc B có state hợp lệ chưa stale.
- `R_AB(t)`: pair đã calibration và READY.
- `U_A(t)`: output unknown cho A tại `t`.

### 3.2 Score

- `p3_A(t)`: probability từ XGBoost C3 specialist cho actor A.
- `p2_AB(t)`: XGBoost C2 evidence mạnh nhất của unordered pair `{A,B}`.
- `H_AB(t)`: directed head/body feature score của A hướng về B. Chỉ dùng landmark head, shoulders, hips, peer position và neutral baseline. Không dùng hand.
- `K_AB(t)`: symmetric hand-interaction feature score của pair `{A,B}`. Dùng hand quality, hand trajectory, hand-to-hand/midpoint geometry. Không bị head feature veto.
- `Q_A(t)`: reliability của head/body feature A, range `[0,1]`.
- `Q^hand_AB(t)`: reliability của pair-hand feature, range `[0,1]`.

### 3.3 Threshold và duration

- `tau_3`: XGBoost C3 threshold. Giá trị artifact hiện tại `0.9580807089805603`; Stage A giữ nguyên.
- `tau_2`: XGBoost C2 threshold từ artifact; Stage A giữ nguyên.
- `tau_H`: minimum head/body feature score cho directed C3. Không chọn bằng cảm tính; lấy từ train-only/dev-camera calibration ở Stage B/C.
- `tau_K`: minimum hand/pair feature score cho C2. Không chọn bằng cảm tính.
- `tau_Q`: minimum feature reliability. Thiếu reliability trả unknown.
- `T_cal = 2000 ms`: neutral calibration duration.
- `T_stale = 500 ms`: thời gian tối đa dùng peer state cũ khi cùng track chưa expired.
- `T_turn = 2000 ms`: mỗi lần quay được giữ ít nhất 2 giây trong demo.
- `L_max = 2000 ms`: latency demo ban đầu; đo từ annotated turn onset đến first C3 flag. Dùng timestamp, không frame count.
- `N_trial = 3`: số lần quay mỗi hướng.
- `T_neutral = 10000 ms`: neutral hard-negative camera interval.

### 3.4 Predicate mục tiêu

C3 cho directed edge:

```text
C3_A(t) = O_AB(t)
          AND R_AB(t)
          AND Q_A(t) >= tau_Q
          AND H_AB(t) >= tau_H
          AND p3_A(t) >= tau_3
```

Hand không xuất hiện trong predicate C3. Hand có thể được log hoặc dùng làm feature XGBoost đã huấn luyện, nhưng không được veto khi predicate trên đạt.

C2 cho unordered pair:

```text
C2_AB(t) = O_AB(t)
           AND Q^hand_AB(t) >= tau_Q
           AND K_AB(t) >= tau_K
           AND p2_AB(t) >= tau_2
```

Resolver tại `t`:

```text
if peer/state không đủ reliability: unknown
else if C2_AB: A=C2, B=C2
else: evaluate C3_A và C3_B độc lập
```

## 4. Evidence hiện tại

- Offline artifact: actor macro-F1 `0.728494623655914`; C3 recall `5/8 = 0.625`.
- Camera mặc định load cùng XGBoost artifact và `tau_3`.
- Camera từng có `p3=0.96009 > tau_3` nhưng hard gate chặn.
- Run khác có `0/338` positive; max `p3=0.8842 < tau_3`.
- Session hai actor mới nhất kết thúc với C3 score khoảng `0.6399` và `0.8115`, cả hai C5.
- Runtime hiện yêu cầu hand quality và quiet hand/finger để mở C3 gate. Đây sai contract mới.
- Camera output hiện thiếu per-frame gate terms; chưa có deterministic red replay.

Evidence trên chỉ đủ chứng minh nhiều failure mode khả dĩ. Chưa đủ chọn một root cause duy nhất.

## 5. Stage A — Observe và tạo red replay

### A1. Khóa provenance mỗi session

Ghi một `session_manifest.json` gồm:

- command đầy đủ và working directory;
- Git commit, branch, dirty file list;
- SHA256 metrics, feature schema và UBJ;
- mọi runtime argument, gồm threshold override;
- camera index, requested width/height/FPS;
- measured FPS, latency p50/p95;
- wall-clock start/end;
- session ID và đường dẫn video/trace.

**Completion criterion:** một session có thể xác định chính xác code, artifact và arguments đã tạo nó. Không field bắt buộc nào rỗng.

### A2. Ghi per-frame trace

Mỗi processed frame, mỗi directed edge, ghi JSONL:

- `timestamp_ms`, `frame_index`, measured inter-frame duration;
- actor ID, peer ID, track present/missed/age;
- actor/peer bbox và pose/hand validity;
- `peer_age_ms`, `peer_stale`, `O_A`, `O_AB`, `R_AB`;
- neutral baseline age, READY state, reset reason;
- `p3_A`, `H_AB`, `Q_A`, `tau_3`, `tau_H`;
- `p2_AB`, `K_AB`, `Q_hand_AB`, `tau_2`, `tau_K`;
- từng legacy C3 gate term và final legacy gate boolean;
- resolver candidate, emitted class, unknown reason;
- first-flag timestamp và per-stage latency.

Trace ghi raw scalar, không chỉ boolean. Trace không chứa frame tương lai.

**Completion criterion:** từ JSONL alone, agent chỉ ra chính xác predicate nào chặn từng expected C3 frame.

### A3. Capture protocol

Một camera session, hai actor, cùng ánh sáng/vị trí:

1. 2 giây neutral calibration; cả hai nhìn trước.
2. 10 giây neutral hard negative.
3. A quay về B, giữ 2 giây, trở neutral; lặp 3 lần.
4. B quay về A, giữ 2 giây, trở neutral; lặp 3 lần.
5. Một lần che/mất peer dưới 500 ms.
6. Một lần che/mất peer trên 500 ms.
7. Một lần cố ý tạo track reset nếu tái hiện an toàn.

Ghi video gốc, trace JSONL và event annotation gồm `neutral`, `turn_onset`, `turn_end`, `occlusion_onset`, `occlusion_end`. Annotation dùng timestamp video.

**Completion criterion:** đủ 6 positive trials, neutral interval và hai peer-loss cases; video/trace/event timestamps đồng bộ.

### A4. Replay command

Tạo một CLI hẹp, ví dụ:

```powershell
python -m pose_gaze.holistic.debug.replay_c3_camera \
  --session-manifest <manifest.json> \
  --trace <frames.jsonl> \
  --events <events.json> \
  --assert-expected
```

Replay đi qua `CausalLiveActorClassifier.update_tracks()` hoặc seam tương đương của production resolver. Không mock score/gate trong end-to-end replay.

Output summary theo trial:

- source actor, peer actor;
- first flag latency;
- maximum `p3`, maximum `H`;
- blocker counts: `not_ready`, `actor_unobservable`, `peer_unobservable`, `score_low`, `head_low`, `legacy_gate_veto`, `resolver_override`, `track_reset`;
- peer false flag và neutral false flag.

**Completion criterion:** command chạy deterministic, red trên camera failure, dưới 30 giây, exit non-zero khi expected C3 không flag.

## 6. Stage A tests — chỉ test có signal

### T-A1: Trace completeness

- **Yêu cầu:** chạy 20-frame fixture qua trace writer.
- **Mong muốn:** đúng một record cho mỗi processed directed edge; đủ mọi field A2; timestamp tăng; không secret/path ngoài session manifest.
- **Lý do:** thiếu một gate scalar làm diagnosis quay lại đoán mò.

### T-A2: Replay catches exact symptom

- **Yêu cầu:** chạy capture C3 fail thật với `--assert-expected`.
- **Mong muốn:** exit non-zero; report trial source không có first flag và blocker cụ thể.
- **Lý do:** đây là feedback loop của bug, không phải test “code chạy không crash”.

### T-A3: Replay determinism

- **Yêu cầu:** chạy cùng capture ba lần.
- **Mong muốn:** verdict, first-flag timestamps và blocker counts giống hoàn toàn.
- **Lý do:** fix không thể đánh giá trên loop đổi kết quả.

### T-A4: Causality

- **Yêu cầu:** replay prefix tới timestamp `t`, sau đó replay cùng prefix với future frames nối thêm.
- **Mong muốn:** mọi output tại hoặc trước `t` giống byte-for-byte.
- **Lý do:** live model không được dùng tương lai.

Không chạy full repository suite tại Stage A. Nó không bắt exact camera symptom.

## 7. Stage B — Repair runtime semantics

Thực hiện từng thay đổi; sau mỗi thay đổi chạy T-A2 và ghi blocker distribution mới.

### B1. Directed pair construction

- Khi đúng hai stable tracks: tạo `A:B`, `B:A`.
- Tính direction bằng peer bbox/anchor thực.
- Xóa fallback “peer nằm bên phải”.
- Pair mất reliability: unknown.

**Completion criterion:** hoán đổi vị trí A/B làm đổi đúng dấu `H_AB`; chỉ source quay về peer đạt directed feature.

### B2. Timestamp state

- Neutral calibration đủ `T_cal` theo timestamp.
- Peer cache hợp lệ tối đa `T_stale` nếu cùng track.
- Velocity chia elapsed milliseconds/seconds, không frame-index delta.
- Window có duration bound; FPS thay đổi không đổi ý nghĩa feature.

**Completion criterion:** cùng motion replay ở 10 FPS và 30 FPS cho cùng qualitative decision và latency gần nhau theo wall-clock.

### B3. Baseline và reset

- READY chỉ khi cả hai actor liên tục observable đủ 2 giây.
- Track ID đổi, pair đổi, gap quá stale: reset calibration.
- UI/trace phân biệt `CALIBRATING`, `READY`, `UNKNOWN`.

**Completion criterion:** không có C3 trước READY; track reset không tái dùng baseline/evidence cũ.

### B4. C3 head override

- Tách `H_AB`, `Q_A` khỏi hand features.
- Xóa hand quality/motion khỏi hard C3 predicate.
- Giữ legacy gate terms trong trace một release để chứng minh veto đã biến mất.
- Chọn `tau_H` trên train-only hoặc labeled dev-camera data; ghi data hash và objective.

**Completion criterion:** fixture có head/body đủ mạnh và hand missing/moving vẫn C3 nếu `p3 >= tau_3`; head quay sai hướng không C3.

### B5. C2 hand override và resolver

- Tính `K_AB` đối xứng cho pair.
- Khi C2 predicate đạt, cả A/B nhận C2; C2 thắng directed C3.
- Head low không veto C2.

**Completion criterion:** hand exchange đạt C2 dù head neutral; head turn không có hand interaction không tạo C2.

## 8. Stage B tests

### T-B1: Directed C3

- **Yêu cầu:** synthetic geometry A nhìn B; B neutral; lặp lại khi đổi phía màn hình.
- **Mong muốn:** chỉ `H_AB` và C3_A đạt; kết quả không phụ thuộc B nằm trái hay phải.
- **Lý do:** khóa contract directed, bắt screen-right fallback.

### T-B2: Head overrides hand veto

- **Yêu cầu:** cùng head/body evidence, ba biến thể hand: valid-still, valid-moving, missing.
- **Mong muốn:** C3 verdict giống nhau khi `Q_A`, `H_AB`, `p3_A` đạt.
- **Lý do:** bắt exact hard-gate false negative hiện tại.

### T-B3: C2 overrides head

- **Yêu cầu:** pair hand evidence đạt C2 trong khi một actor cũng đạt C3.
- **Mong muốn:** cả pair C2.
- **Lý do:** khóa resolver `C2 > C3` và symmetric C2.

### T-B4: Peer staleness

- **Yêu cầu:** peer vắng 400 ms rồi 600 ms, cùng track state.
- **Mong muốn:** 400 ms dùng cached peer với `peer_stale=true`; 600 ms unknown.
- **Lý do:** kiểm tra causal history có bound, không dùng stale geometry vô hạn.

### T-B5: Track reset

- **Yêu cầu:** đổi peer `student_02` thành `student_03` giữa stream.
- **Mong muốn:** pair trở CALIBRATING; không carry baseline, latch hoặc first flag.
- **Lý do:** chống evidence leakage giữa identity.

### T-B6: FPS invariance

- **Yêu cầu:** resample cùng capture ở 10 FPS và 30 FPS, giữ timestamps.
- **Mong muốn:** cùng actor/class; latency sai khác trong tolerance được ghi trước test.
- **Lý do:** camera thực chỉ khoảng 10 FPS; frame-count logic tạo domain mismatch.

### T-B7: Original camera replay

- **Yêu cầu:** chạy lại T-A2 sau repair.
- **Mong muốn:** 6/6 source trials flag trong `L_max`; 0 peer false flag; 0 neutral C3; peer mất quá 500 ms trả unknown.
- **Lý do:** unit tests không thay thế exact reported failure.

Chỉ chạy focused tests cho files/seams đã đổi và original camera replay. Chạy broader holistic tests sau khi T-B7 xanh, nhằm bắt regression chứ không dùng làm bằng chứng camera quality.

## 9. Stage C — Retrain XGBoost, chỉ khi cần

Kích hoạt Stage C khi:

- Stage B semantics/tests đúng;
- capture vẫn fail vì `p3 < tau_3` hoặc head score không tách C3/C5;
- trace chứng minh tracking/observability đủ.

### C1. Data

- Thu actor/edge-labeled camera traces cho C3 và C5 hard negatives.
- Key tối thiểu: `(session, video, actor_id, peer_actor_id, timestamp)`.
- Split theo actor pair/session; không cho cùng identity/session qua train và test.
- Giữ full stream; action interval chỉ làm truth/evaluation, không crop model input.

### C2. Features và model

- C3 sample là ordered edge `A:B`, không actor-only class row.
- C2 sample là unordered pair `{A,B}`.
- Reliability/missingness là explicit features; missing peer không trở thành C5 negative.
- So sánh current XGBoost với retrained XGBoost trên cùng splits, features, replay và budgets.

### C3. Calibration

- Fit `tau_3`, `tau_H`, duration/release policy trên train-only OOF replay.
- Objective gồm directed C3 recall, neutral false alerts, peer false flags, unknown coverage và first-flag latency.
- Không chọn threshold bằng locked test hoặc camera promotion session.

**Completion criterion:** retrained model thắng current artifact trên same replay, không tăng false-alert budget, provenance hashes đầy đủ.

## 10. Stage D — Camera promotion gate

Demo gate ban đầu:

- `N_trial=3` mỗi hướng; tổng 6 directed C3 trials.
- 6/6 source flag trong `L_max=2000 ms` sau turn onset.
- 0/6 neutral peer bị C3.
- 0 C3 trong `T_neutral=10000 ms`.
- Peer stale quá 500 ms: 100% unknown, 0 C5/C3.
- Không crash, không non-monotonic timestamp, không silent track-state reuse.
- Báo measured FPS và latency p50/p95; không suy FPS từ camera setting.
- Replay artifact và live run cho cùng verdict trên saved capture.

Promotion cuối cần thêm locked actor-level test theo `(video, actor_id)` và nhiều camera sessions/actors. Demo gate chỉ chứng minh simple two-actor scenario; không chứng minh production generalization.

## 11. Handoff cho agent thực hiện

Agent bắt đầu từ Stage A. Trước mỗi stage:

1. Đọc `AGENTS.md` và xác nhận checkout `C:\Real-Time-Exam-Proctoring-System`, branch `new-test-pipieline`.
2. Giữ nguyên user changes; liệt kê dirty files liên quan trước edit.
3. Không train hoặc đổi thresholds trong Stage A/B nếu trace chưa chứng minh cần.
4. Mỗi test báo: command, fixture, expected, actual, verdict và artifact path.
5. Dừng tại completion criterion của stage; không tự chạy stage sau.

Stage A hoàn thành chỉ khi có một deterministic red replay từ exact real-camera failure. Stage B hoàn thành chỉ khi replay đó xanh và focused semantic tests xanh.
