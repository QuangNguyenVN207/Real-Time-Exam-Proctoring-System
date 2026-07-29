# Paper Tracking for the Real-Time Exam Proctoring System

## 1. Mục tiêu

Module paper tracking giải quyết tình trạng `object_detect` nhầm lẫn giữa:

- `test_paper`: giấy thi hợp lệ;
- `cheat_sheet`: tài liệu không được phép;
- các vật thể có hình dạng gần giống giấy hoặc điện thoại.

Không nên quyết định gian lận chỉ từ nhãn YOLO của một frame. Hệ thống mới kết hợp:

1. phát hiện người và gán `track_id`;
2. phát hiện toàn bộ bounding box giấy;
3. gán `paper_id` ổn định cho từng tờ giấy;
4. liên kết giấy với `owner_track_id` của học sinh;
5. đăng ký giấy thi hợp lệ trong giai đoạn chuẩn bị;
6. chỉ cảnh báo khi một tờ giấy lạ tồn tại đủ lâu hoặc thay thế giấy đã đăng ký.

## 2. Giới hạn của checkpoint hiện tại

Checkpoint `weights/yolov8_finetuned.pt` hiện tại chỉ có bốn lớp và chưa có lớp
`test_paper`. Vì vậy, model chưa thực sự học được ranh giới giữa giấy thi và
cheat sheet.

Với checkpoint này:

- detection có hình dạng giấy được chuyển thành `paper_unknown`;
- không phát cảnh báo `cheat_sheet` trực tiếp từ một detection đơn lẻ;
- paper tracker sử dụng vị trí, lịch sử theo thời gian và số lượng giấy để đưa ra
  quyết định;
- tờ giấy đầu tiên ổn định của mỗi học sinh có thể được đăng ký là giấy thi;
- tờ giấy vật lý thứ hai được giữ bằng một `paper_id` khác và được kiểm tra như
  giấy lạ.

Khi có checkpoint sáu lớp:

```text
smartwatch
earphone
cheat_sheet
smartphone
calculator
test_paper
```

hệ thống vẫn dùng tracking và label smoothing. Một frame dự đoán sai không được
phép đổi ngay trạng thái của cả track.

## 3. Kiến trúc

```text
Camera frame
    |
    +--> Person detector (yolov8n.pt)
    |       |
    |       +--> temporary track_id
    |               |
    |               +--> proctor assigns stable person_id
    |
    +--> Custom object detector (yolov8_finetuned.pt)
            |
            +--> prohibited object detections
            |
            +--> all paper detections
                     |
                     v
              PaperTracker
              - paper_id
              - label smoothing
              - owner_track_id
              - registration
              - suspicious-paper rules
                     |
                     v
            alerts + evidence + API state
```

`PoseGazePaperPipeline` là lớp kết hợp person tracking, object detection và paper
tracking. Person detector phải dùng model COCO `yolov8n.pt`; checkpoint custom chỉ
dùng cho các vật thể của bài toán giám sát thi.

### ID người do giám thị gán

`track_id` là handle số nội bộ mà tracker tạo để ghép bounding box giữa các frame.
Đây không phải danh tính của thí sinh. Khi webcam phát hiện một người chưa có ID,
giám thị nhập `person_id` (ví dụ `SV001`) trực tiếp trong cửa sổ camera. Kết quả/API vẫn giữ
`track_id` để tương thích, nhưng UI và các module nghiệp vụ phải dùng `person_id`
làm định danh chính.

Tracker ghép người bằng IoU, khoảng cách tâm và fingerprint ngoại hình. Ngay khi
giám thị gán `person_id`, pipeline lưu fingerprint kết hợp vùng mặt, cấu trúc
cạnh và màu áo. Kho fingerprint chỉ nhận người đã được xác nhận, không học các
track tạm.

Nếu người mất dấu đủ lâu rồi quay lại ở vị trí khác, fingerprint đạt ngưỡng sẽ
tự động:

1. khôi phục numeric `track_id` ban đầu;
2. giữ nguyên `person_id` do giám thị đã gán;
3. chuyển paper ownership và đăng ký giấy về track đã khôi phục.

Người có fingerprint khác không được kế thừa ID chỉ vì ngồi đúng ghế cũ; hệ
thống tạo `UNASSIGNED temp_track=...` để giám thị xác nhận. Nếu mặt/thân bị che,
crop quá nhỏ hoặc ánh sáng quá kém nên không tạo được fingerprint, có thể nhập
lại đúng `person_id` cũ như cơ chế fallback thủ công.

## 4. Các file liên quan

### File đã sửa

| File | Vai trò |
|---|---|
| `backend/ai_services/object_detect/object_detect.py` | Giữ toàn bộ paper box, tách paper khỏi đường cảnh báo trực tiếp và trả `paper_detections`. |
| `backend/ai_services/object_detect/test_webcam.py` | Hiển thị paper candidate và các vật thể cấm theo contract mới. |
| `backend/ai_services/pose_gaze/tracking/manager.py` | Quản lý person tracker, paper tracker, đăng ký giấy và bật giám sát. |
| `backend/ai_services/pose_gaze/tracking/schemas.py` | Bổ sung thao tác tâm box, đường chéo và mở rộng bounding box. |
| `backend/ai_services/pose_gaze/__init__.py` | Export các thành phần paper tracking. |
| `backend/ai_services/pose_gaze/README.md` | Hướng dẫn chạy và liên kết tới tài liệu này. |
| `backend/api/pose_gaze_routes.py` | API đọc trạng thái giấy, đăng ký và bật/tắt monitoring. |
| `backend/core/config.py` | Các ngưỡng paper tracking và cấu hình cảnh báo. |

### File mới

| File | Vai trò |
|---|---|
| `backend/ai_services/pose_gaze/tracking/paper_tracking.py` | Logic tạo `paper_id`, association, label smoothing, owner assignment và cảnh báo. |
| `backend/ai_services/pose_gaze/paper_pipeline.py` | Pipeline kết hợp pose/gaze, person tracking và paper tracking. |
| `backend/ai_services/object_detect/test_paper_tracking_webcam.py` | Demo end-to-end bằng webcam. |
| `backend/ai_services/pose_gaze/PAPER_TRACKING.md` | Tài liệu kỹ thuật. |
| `backend/ai_services/object_detect/tests/test_object_detect.py` | Kiểm tra contract của object detector. |
| `backend/ai_services/pose_gaze/tracking/tests/test_paper_tracking.py` | Unit test cho paper tracker. |
| `backend/ai_services/pose_gaze/tracking/tests/test_paper_pipeline.py` | Integration test cho pipeline. |

## 5. Contract của object detector

Object detector không còn chỉ giữ bounding box tốt nhất của mỗi class. Kết quả
phải giữ mọi instance vì hai tờ giấy khác nhau cần tạo hai track khác nhau.

Kết quả logic gồm hai nhóm:

```python
{
    "detections": [...],        # điện thoại, tai nghe, smartwatch, ...
    "paper_detections": [...]   # tất cả paper candidates
}
```

Mỗi paper detection cần tối thiểu:

```python
{
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.91,
    "raw_label": "paper_unknown"
}
```

Không cộng trực tiếp `paper_unknown` vào bộ đếm gian lận cũ.

## 6. Paper ID và association theo thời gian

Mỗi tờ giấy vật lý được biểu diễn bởi một track:

```text
PaperTrack
├── paper_id
├── paper_id_source
├── bbox
├── raw_label_history
├── stable_label
├── owner_track_id
├── hit_count
├── missed_count
├── status
└── suspicious_reasons
```

Ở mỗi lần inference:

1. lấy toàn bộ paper detections mới;
2. so khớp chúng với paper tracks hiện có bằng độ chồng lấp và khoảng cách tâm;
3. cập nhật track đã khớp;
4. tạo ID track tạm cho detection không khớp;
5. giữ track qua một số frame bị che khuất ngắn;
6. xóa track chỉ khi vượt quá ngưỡng mất dấu.

ID tạm chỉ phục vụ association, không phải danh tính của tờ giấy. Khi cửa sổ
camera hỏi, giám thị nhập một `paper_id` số dương cố định, ví dụ `101`. Hệ thống remap
track tạm sang ID này và đánh dấu `paper_id_source=manual`. Nếu giấy mất dấu lâu,
pipeline dùng crop của tờ giấy để tạo fingerprint gồm bố cục tần số thấp và cấu
trúc cạnh/chữ. Fingerprint được đăng ký theo cặp `(owner_person_id, paper_id)`.

Khi giấy xuất hiện lại:

1. fingerprint khớp và owner khớp → tự khôi phục `paper_id=101` cùng
   `authorized_exam_paper`, không hỏi nhập lại;
2. fingerprint khác dù box nằm đúng vị trí cũ → tạo paper track mới;
3. paper mới được gắn với `owner_person_id` theo vị trí;
4. vì mỗi học sinh chỉ có một đề thi authorized, paper mới phát
   `additional_paper` hoặc `paper_replacement` sau ngưỡng xác nhận.

Nhờ vậy, nếu YOLO đổi nhãn hoặc tờ giấy di chuyển giữa các frame thì ID đã gán
vẫn được giữ nguyên. API trả `paper_id_assigned=false` trong thời gian giấy mới
chỉ có ID tạm; UI không nên coi số tạm đó là ID chính thức. Trường
`appearance_identity_registered=true` cho biết fingerprint của đề thi đã được
ghi nhớ thành công.

## 7. Liên kết giấy với học sinh

Mỗi học sinh được giám thị gán một `person_id` ổn định trên track đang hiển thị.
Paper tracker vẫn dùng `owner_track_id` nội bộ để liên kết theo không gian, đồng
thời API trả `owner_person_id` để các module nghiệp vụ không phụ thuộc ID tạm:

1. tâm tờ giấy nằm trong person box đã mở rộng;
2. giấy có độ chồng lấp hợp lý với vùng của người;
3. nếu cần, chọn người gần nhất theo khoảng cách tâm đã chuẩn hóa.

Ví dụ:

```text
student person_id=SV001 (internal track_id=1)
├── paper_id=1 -> authorized_exam_paper
└── paper_id=3 -> suspicious, reason=additional_paper

student person_id=SV002 (internal track_id=2)
└── paper_id=2 -> authorized_exam_paper
```

Danh sách giấy hợp lệ được quản lý theo từng học sinh, không dùng một giấy hợp lệ
chung cho toàn bộ camera.

## 8. Giai đoạn SETUP và MONITORING

### SETUP

Trong giai đoạn chuẩn bị:

- đặt giấy thi hợp lệ trước mặt học sinh;
- chờ tờ giấy được detect ổn định;
- hệ thống gán `paper_id`;
- tờ đầu tiên ổn định của học sinh được đăng ký;
- trạng thái chuyển thành `authorized_exam_paper`;
- bounding box hiển thị màu xanh.

### MONITORING

Sau khi người dùng arm hệ thống:

- không tự động đăng ký tờ giấy mới;
- giấy đã đăng ký tiếp tục giữ màu xanh;
- một `paper_id` bổ sung được theo dõi qua nhiều lần inference;
- chỉ khi đủ ổn định, giấy bổ sung mới chuyển sang `suspicious`;
- alert được phát kèm ảnh bằng chứng và thông tin track.

Không nên arm khi giấy thi ban đầu chưa chuyển xanh.

## 9. Trạng thái của một paper track

| Trạng thái | Ý nghĩa |
|---|---|
| `candidate` | Giấy vừa xuất hiện, chưa đủ lịch sử để quyết định. |
| `authorized_exam_paper` | Giấy thi đã đăng ký cho học sinh. |
| `observed` | Track hợp lệ đang được theo dõi nhưng chưa cần cảnh báo. |
| `suspicious` | Giấy lạ đã vượt qua ngưỡng xác nhận. |
| `lost` | Track tạm thời không được nhìn thấy nhưng chưa bị xóa. |

Các lý do cảnh báo quan trọng:

| Reason | Trường hợp |
|---|---|
| `additional_paper` | Học sinh đã có giấy hợp lệ nhưng xuất hiện thêm một `paper_id`. |
| `paper_replacement` | Giấy hợp lệ biến mất và một tờ giấy có ID khác xuất hiện thay thế. |
| `stable_cheat_sheet_label` | Với model sáu lớp, lịch sử nhãn nghiêng ổn định về `cheat_sheet`. |

## 10. Làm mượt nhãn

Không dùng nhãn của frame cuối cùng làm kết luận. Mỗi track lưu lịch sử:

```text
frame 1: test_paper
frame 2: test_paper
frame 3: cheat_sheet
frame 4: test_paper
```

Trong ví dụ này, track vẫn được xem là `test_paper`; một frame xấu không làm nhãn
đổi sang `cheat_sheet`.

Ngược lại, `cheat_sheet` chỉ được xác nhận khi:

- cùng một `paper_id` tồn tại đủ lâu;
- nhãn `cheat_sheet` chiếm ưu thế trong cửa sổ lịch sử;
- độ tin cậy và số lần quan sát vượt ngưỡng cấu hình;
- track không thuộc tập giấy thi đã đăng ký.

## 11. Quy tắc cho checkpoint bốn lớp

Vì chưa có `test_paper`, hệ thống áp dụng quy tắc:

```text
paper detection
      |
      v
paper_unknown
      |
      +--> first stable paper during SETUP
      |        -> authorized_exam_paper
      |
      +--> new paper_id during MONITORING
               -> candidate
               -> suspicious after confirmation threshold
```

Điều này giúp giảm nhầm giấy thi thành cheat sheet, nhưng không thay thế việc
fine-tune model sáu lớp.

## 12. REST API

Các route trong `backend/api/pose_gaze_routes.py` cung cấp các thao tác:

1. tạo session, gửi/đọc person tracks;
2. gán hoặc bỏ `person_id` trên một track đang hiển thị;
3. gửi/đọc paper tracks;
4. gán `paper_id` cố định cho paper track tạm;
5. đăng ký hoặc bỏ đăng ký paper làm giấy thi hợp lệ;
6. arm/disarm chế độ monitoring.

Gán ID cho người (payload `student_id` cũ vẫn được chấp nhận để tương thích):

```http
PUT /api/pose-gaze/sessions/exam-room-01/tracks/1/assignment
Content-Type: application/json

{"person_id": "SV001"}
```

Gán ID giấy cố định cho paper track tạm `3`:

```http
PUT /api/pose-gaze/sessions/exam-room-01/papers/3/identity
Content-Type: application/json

{"stable_paper_id": 101}
```

Thông tin paper trả về cần có dạng tương đương:

```json
{
  "paper_id": 101,
  "paper_id_assigned": true,
  "paper_id_source": "manual",
  "owner_track_id": 1,
  "owner_person_id": "SV001",
  "label": "paper_unknown",
  "status": "suspicious",
  "bbox": [310, 220, 490, 410],
  "reasons": ["additional_paper"]
}
```

Thông tin alert cần chứa:

```json
{
  "type": "possible_cheat_sheet",
  "owner_track_id": 1,
  "owner_person_id": "SV001",
  "paper_id": 101,
  "reasons": ["additional_paper"],
  "evidence_path": "..."
}
```

Prefix URL thực tế phụ thuộc cách router được mount trong ứng dụng. Khi tích hợp
frontend, lấy đường dẫn trực tiếp từ `backend/api/pose_gaze_routes.py`.

## 13. Cấu hình

Các ngưỡng được đặt trong `backend/core/config.py`, không hard-code trong webcam
demo:

- confidence tối thiểu của paper detection;
- ngưỡng association theo IoU;
- khoảng cách tâm tối đa khi ghép track;
- số lần quan sát tối thiểu trước khi đăng ký;
- số lần quan sát tờ giấy lạ trước khi cảnh báo;
- số frame cho phép mất dấu;
- ngưỡng cosine của appearance fingerprint khi re-identify người;
- ngưỡng cosine của appearance fingerprint khi re-identify giấy;
- kích thước cửa sổ label smoothing;
- ngưỡng xác nhận `test_paper` hoặc `cheat_sheet`;
- hệ số mở rộng person box khi tìm owner;
- thời gian cooldown giữa hai alert giống nhau.

Nếu camera có FPS cao nhưng object detector chỉ chạy mỗi vài frame, ngưỡng xác
nhận phải được hiểu theo số lần inference, không phải số frame camera.

## 14. Chạy webcam demo

Từ thư mục gốc dự án:

```powershell
python -m backend.ai_services.object_detect.test_paper_tracking_webcam
```

Quy trình kiểm tra:

1. Giữ focus ở cửa sổ camera, gõ ID cố định cho người đang được khoanh (ví dụ
   `SV001`) rồi nhấn `Enter`; chờ label người hiện `memory=on`.
2. Khi camera hỏi paper, gõ ID số dương cố định (ví dụ `101`) rồi `Enter`; trước khi
   nhập, box chỉ hiện `UNASSIGNED temp_paper=...`.
3. Chỉ đặt giấy thi hợp lệ trước camera.
4. Chờ person và paper box đều hiện `memory=on`, paper chuyển xanh.
5. Nhấn `A` để arm/khóa đăng ký. Demo từ chối arm nếu fingerprint chưa được ghi
   nhớ; cần chỉnh góc camera/ánh sáng để nhìn rõ nội dung giấy.
6. Đưa một tờ giấy vật lý khác vào và gán một `paper_id` khác.
7. Sau ngưỡng xác nhận, tờ mới chuyển đỏ và tạo alert.
8. Nhấn `Q` hoặc `Esc` để thoát.

Trong lúc nhập: `Backspace` để sửa và `Esc` để bỏ qua detection tạm. Khi không
có prompt, `Q` hoặc `Esc` mới là thoát. Demo không cho nhấn `A` thành công nếu
vẫn còn person/paper chưa được gán.

Nếu người đã có `memory=on` rời camera lâu rồi quay lại, hệ thống tự khôi phục
`person_id` và numeric `track_id` ban đầu. Chỉ khi fingerprint không đủ rõ hoặc
không đạt ngưỡng, cửa sổ mới hỏi lại; lúc đó nhập chính `person_id` cũ.
Nếu đề thi quay lại và fingerprint đủ rõ, hệ thống tự khôi phục ID, authorization
và owner. Chỉ cần nhập lại `paper_id` khi crop quá mờ/trắng hoặc góc nhìn thay đổi
quá lớn khiến fingerprint không đạt ngưỡng.

Màn hình demo nên hiển thị:

```text
person_id (do giám thị gán)
temporary track_id (chỉ hiện khi chưa gán)
paper_id (do giám thị gán)
temporary paper ID (chỉ hiện khi chưa gán)
owner_track_id
owner_person_id
stable label
paper status
monitoring state
alert reason
```

## 15. Chạy test

Chạy các test liên quan:

```powershell
python -m unittest discover backend/ai_services/object_detect/tests
python -m unittest discover backend/ai_services/pose_gaze/tracking/tests
```

Các trường hợp cần được bảo vệ bởi test:

- giữ `paper_id` khi box di chuyển nhẹ;
- gán ID thủ công thay cho temporary paper ID;
- khôi phục `paper_id` và authorization cũ sau khi giấy bị re-track;
- tự nhận đề thi cũ bằng appearance fingerprint;
- không cho giấy khác ở cùng vị trí kế thừa ID đề thi;
- alert giấy khác phải chứa đúng `owner_person_id`;
- giữ person `track_id` khi box di chuyển nhưng không còn chồng lấp;
- tự khôi phục `person_id` sau khi người rời khung hình rồi quay lại;
- không cho người khác ngồi cùng vị trí kế thừa `person_id`;
- khôi phục person `track_id` cũ khi gán lại cùng `person_id`;
- không đổi ID khi nhãn YOLO dao động;
- giữ track qua che khuất ngắn;
- tạo ID khác cho tờ giấy vật lý thứ hai;
- mỗi học sinh có tập giấy hợp lệ riêng;
- tờ đầu tiên trong SETUP được đăng ký;
- tờ bổ sung không cảnh báo ngay ở frame đầu;
- tờ bổ sung chuyển `suspicious` sau ngưỡng;
- phát hiện `paper_replacement`;
- một frame nhãn `cheat_sheet` sai không tạo alert;
- pipeline trả đúng owner, status và alert.

### Chạy regression test với MP4

Hai script tương tác nhận cả webcam index lẫn đường dẫn video:

```powershell
python -m backend.ai_services.object_detect.test_webcam `
  --source data/smartphone.mp4

python -m backend.ai_services.object_detect.test_paper_tracking_webcam `
  --source data/cheatsheet.mp4
```

Để chạy headless cả raw object detection, person identity và paper tracking,
đồng thời xuất `report.json`, evidence và video đã annotate:

```powershell
python -m backend.ai_services.object_detect.test_video_scenarios `
  data/smartphone.mp4 data/cheatsheet.mp4 `
  --frame-stride 3 --setup-seconds 5
```

`--setup-seconds` phải kết thúc trước khi vật cấm xuất hiện. Nếu detector chưa
nhìn thấy đề thi hợp lệ trong giai đoạn SETUP thì video đó không đủ điều kiện để
đánh giá paper re-ID; mọi paper xuất hiện sau ARM đều chỉ là unregistered paper.

## 16. Hạn chế

1. Nếu hai tờ giấy chồng hoàn toàn và YOLO chỉ trả một bounding box, tracker
   không thể biết có hai tờ.
2. Fingerprint cần nhìn thấy đủ chữ/hình/bố cục trên giấy. Giấy trắng, ảnh quá mờ,
   bị che nhiều hoặc thay đổi góc nhìn rất lớn không đủ thông tin để tự xác thực;
   khi đó hệ thống giữ ID tạm và yêu cầu giám thị xác minh/gán lại.
3. Person appearance re-ID là bộ nhớ trong cùng session, dựa trên vùng mặt và
   quần áo nhìn thấy từ camera; đây không phải xác thực sinh trắc học. Che mặt,
   thay áo, ánh sáng/góc nhìn thay đổi rất lớn hoặc hai người quá giống nhau có
   thể khiến hệ thống giữ ID tạm và yêu cầu giám thị nhập lại `person_id`.
4. Mặc định hiện tại là một tờ giấy hợp lệ cho mỗi học sinh. Nếu kỳ thi cho phép
   nhiều trang rời, đề thi và giấy trả lời riêng, cấu hình phải cho phép nhiều
   `paper_id` hợp lệ.
5. Tracking giảm false positive theo thời gian nhưng không giúp model hiểu nội
   dung trên giấy.
6. Muốn phân biệt thật sự giữa `test_paper` và `cheat_sheet`, vẫn phải fine-tune
   checkpoint sáu lớp bằng dữ liệu đủ đa dạng.

## 17. Khuyến nghị dữ liệu cho model sáu lớp

Dataset nên có hard negatives:

- đề thi có chữ dày, chữ thưa, bảng và hình;
- cheat sheet viết tay hoặc in, nhiều kích thước;
- calculator, hộp bút, ví, sách và vật có nắp;
- smartphone ở nhiều góc, màn hình sáng/tắt;
- giấy bị gấp, che một phần hoặc đặt dưới tay;
- nhiều tờ giấy chồng và tách rời;
- ảnh từ đúng webcam và ánh sáng sẽ dùng khi triển khai.

Không nên chỉ dùng augmentation để tạo khác biệt. Ảnh thật từ bối cảnh phòng thi
quyết định khả năng tổng quát hóa của model.

## 18. Kết luận

Paper tracking không dựa vào một prediction đơn lẻ. Hệ thống nhận dạng từng tờ
theo thời gian, gắn nó với đúng học sinh và so sánh với giấy đã đăng ký:

```text
student track_id
    -> authorized paper_id
    -> any persistent additional paper_id
    -> possible_cheat_sheet alert
```

Cách này xử lý được lỗi nhãn dao động và giảm việc báo giấy thi là cheat sheet.
Checkpoint sáu lớp vẫn là bước cần thiết để đạt độ chính xác cao trong triển
khai thực tế.
