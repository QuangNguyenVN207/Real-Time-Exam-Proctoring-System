# Chức năng của `object_detect` trong `develop-paper-count`

## 1. Mục tiêu tổng thể

Phần `object_detect` thực hiện ba nhiệm vụ liên quan nhưng được tách riêng:

1. Phát hiện vật thể cấm như `smartphone`, `earphone`, `smartwatch`.
2. Phát hiện mọi vùng giấy có thể nhìn thấy, gồm `cheat_sheet`, `test_paper`, `paper_unknown` và `book`.
3. Trong thử nghiệm `develop-paper-count`, quyết định giấy lạ bằng **sự thay đổi số lượng giấy**, không tạo và không theo dõi `paper_id`.

`person_id` vẫn được giữ ổn định bằng hệ thống tracking và appearance memory. Chỉ phần định danh giấy được thay bằng logic đếm.

## 2. Sơ đồ luồng xử lý

```mermaid
flowchart TD
    A[Frame OpenCV] --> B[Person detector]
    B --> C[TrackingManager + appearance fingerprint]
    C --> D[Danh sách người có track_id và person_id]
    D --> E[Tạo ROI quanh thân, tay, bàn và đùi]
    A --> F[Model chính best (1).pt]
    A --> G[YOLOv8n COCO fallback]
    E --> F
    E --> G
    F --> H[Chuẩn hóa và lọc detection]
    G --> H
    H --> I{Loại detection}
    I -->|Điện thoại, tai nghe, đồng hồ| J[Xác nhận theo nhiều inference và vị trí]
    I -->|Giấy, book| K[Gom các bbox trùng thành paper observation]
    K --> L[So sánh count với baseline SETUP]
    L -->|Count không tăng| M[Không có paper alert]
    L -->|Count tăng đủ số inference| N[cheat_sheet NEW]
    D --> N
    N --> O[Gán owner là person_id gần nhất]
    J --> P[Danh sách alert cuối]
    O --> P
```

## 3. Hai lớp detector chính

### 3.1. `ObjectDetector`

File: `backend/ai_services/object_detect/object_detect.py`

Đây là adapter nhỏ dùng cho hợp đồng `process_frame(frame, timestamp)` của server đa module.

#### `BANNED_ITEMS`

Danh sách cố định:

```python
["cheat_sheet", "earphone", "smartwatch", "smartphone"]
```

Chỉ các nhãn thuộc danh sách này và vượt confidence threshold mới được trả về.

#### `__init__(model_path, confidence_threshold=0.5, device=None, model=None)`

- Nạp YOLO một lần vào RAM hoặc VRAM.
- Tự chọn `cuda` nếu PyTorch nhìn thấy GPU, ngược lại dùng `cpu`.
- Cho phép truyền fake model qua tham số `model` để unit test không cần nạp weight thật.
- Tắt log mặc định của Ultralytics.
- Di chuyển model sang thiết bị đã chọn.

#### `process_frame(frame, timestamp)`

- Kiểm tra frame phải là NumPy array hợp lệ và không rỗng.
- Resize frame bắt buộc về `640x640`.
- Chạy YOLO với `verbose=False`.
- Lọc theo `BANNED_ITEMS` và confidence.
- Quy đổi bbox từ ảnh `640x640` về kích thước frame gốc bằng `scale_x` và `scale_y`.
- Lọc một số smartphone giả là calculator hoặc bút.
- Trả về `None` nếu không có vi phạm.
- Toàn bộ hàm nằm trong `try...except`; frame hỏng hoặc model lỗi chỉ làm bỏ qua frame đó, không làm sập worker.

Kết quả thành công có dạng:

```json
{
  "module": "object_detect",
  "status": "alert",
  "timestamp": 1700000000.5,
  "details": {
    "detections": [
      {
        "label": "smartphone",
        "confidence": 0.91,
        "bbox": [150, 200, 300, 450]
      }
    ],
    "model_input_size": [640, 640],
    "original_frame_size": [1280, 720]
  },
  "detections": []
}
```

`detections` được giữ cả trong `details` và top-level để tương thích hợp đồng cũ.

### 3.2. `ObjectDetectModule`

Đây là detector đầy đủ dùng cho video, webcam, person ROI và paper pipeline.

Khác với `ObjectDetector`, lớp này không tự kết luận mọi detection `cheat_sheet` là gian lận. Giấy được chuyển cho policy downstream vì cùng một tờ có thể bị model đổi nhãn giữa `test_paper` và `cheat_sheet` ở các frame khác nhau.

#### Khởi tạo model

- Model chính: `weights/best (1).pt` theo `settings.yolo_model_path`.
- Model phụ: `weights/yolov8n.pt`, pretrained COCO.
- Tự chọn CUDA/CPU.
- Model phụ tìm `cell phone`, `book` và các lớp dễ gây nhầm như `remote`.
- `cell phone` được chuẩn hóa thành `smartphone`.
- `book` được chuẩn hóa thành `cheat_sheet` khi `book_as_cheatsheet_enabled=True`.

Khi fallback bật, smartphone từ model chính bị bỏ. Smartphone từ YOLOv8n pretrained trở thành nguồn chính để giảm lỗi `best (1).pt` nhận bút hoặc background thành điện thoại. Các lớp khác của model chính vẫn được giữ.

#### `process(frame, session_id, frame_id, person_rois=None)`

- Đếm số frame riêng cho từng session.
- Chỉ chạy inference mỗi `N` frame để giảm tải.
- Frame bị bỏ qua nhận bản kết quả cache gần nhất với `inference_ran=False`.
- Frame thật sự chạy model có `inference_ran=True`.
- `requested_frame_id` luôn phản ánh frame mà caller yêu cầu.

Paper-count chỉ tăng debounce counter khi `inference_ran=True`. Detection cache không được xem như một lần phát hiện mới.

#### `_process_sync(...)`

Một lần inference thật gồm:

1. Chạy model chính trên toàn frame ở kích thước cấu hình.
2. Chạy lại các class giấy của model chính trong person ROI để cứu giấy nhỏ hoặc giấy gấp.
3. Chạy YOLOv8n fallback trên toàn frame/tile.
4. Chạy YOLOv8n trong person ROI để phóng lớn vùng tay, bàn và đùi.
5. Chuẩn hóa class và bbox về tọa độ frame gốc.
6. Lọc phone confuser, calculator và bút.
7. NMS các bbox trùng.
8. Xác nhận vật cấm không phải giấy theo nhiều inference.
9. Trả toàn bộ paper candidate cho paper monitor.

#### `_extract_detections(predictions)`

- Đọc class, confidence và `xyxy` từ kết quả YOLO.
- Chỉ giữ class nằm trong `flagged_classes` hoặc `paper_class_names`.
- Dùng threshold riêng cho từng class.
- Gắn các trường:
  - `class_name`
  - `display_name`
  - `confidence`
  - `bbox_xyxy`
  - `is_paper_candidate`
  - `source`

#### `_detect_custom_paper_rois(...)`

- Chạy riêng các class giấy của model chính trên ROI của từng người.
- Inference mặc định ở 768 pixel với threshold thấp hơn toàn frame.
- Chuyển bbox từ tọa độ crop về tọa độ frame.
- Gắn `owner_track_id_hint` và `owner_person_id_hint` theo người tạo ROI.
- Mục tiêu là cứu giấy gấp, giấy nhỏ hoặc giấy nằm trên đùi mà full-frame 640 bỏ sót.

#### `_detect_auxiliary_fallback(...)`

- Dùng YOLOv8n COCO để phát hiện điện thoại và book.
- Frame có cạnh lớn nhất không quá 960 dùng một ảnh toàn frame.
- Frame lớn hơn 960 được chia thành bốn tile chồng lấn để vật nhỏ có nhiều pixel hơn.
- Nếu có person tracking, chạy thêm ROI theo từng người.
- Nếu không có person ROI và video rộng, tạo ba crop dạng cột để tìm book nhỏ.
- Sau cùng lọc confuser và chạy NMS.

#### `_person_roi_specs(...)`

Tạo vùng tương tác quanh mỗi người:

- Mở rộng bbox sang hai bên và phía trên.
- Với người thấy toàn thân, cắt bớt vùng chân.
- Với bbox thân trên, mở rộng nhẹ xuống dưới để chứa tay và bàn.
- Có thể tạo thêm context ROI lớn hơn cho book.
- Tối đa xử lý số người theo `person_roi_max_people`, hiện tại là 2.

#### Chuẩn hóa `book` thành `cheat_sheet`

Có hai lớp bảo đảm việc ánh xạ:

1. `object_class_aliases["book"] = "cheat_sheet"`.
2. Model COCO xác định class ID của `book` và đưa nó vào danh sách paper candidate.

Vì vậy book không đi vào temporal alert của smartphone/earphone. Nó được chuyển sang paper-count và chỉ trở thành cảnh báo nếu làm số giấy vượt baseline.

#### Lọc smartphone nhận nhầm

`_remove_smartphone_confusers()` loại smartphone candidate khi:

- Bbox chồng với class `remote` từ COCO.
- `_looks_like_calculator()` nhìn thấy lưới nút calculator đủ rõ.
- `_looks_like_pen()` nhìn thấy thành phần sáng, hẹp và dài giống bút/marker.

Các bộ lọc này đều nằm trong `try...except`; lỗi OpenCV không làm dừng luồng và mặc định không loại detection.

#### NMS và hình dạng hợp lệ

- `_nms()` loại bbox trùng theo từng class, không loại book chỉ vì nó chồng với smartphone.
- Nếu bbox toàn frame và bbox ROI trùng nhau, owner hint từ ROI được chép sang bbox được giữ.
- `_is_plausible_book_bbox()` bỏ vùng quá mỏng giống cạnh bàn.
- `_is_plausible_roi_phone_bbox()` bỏ bbox quá vuông hoặc quá dài, thường là bút.
- `_bbox_center_inside()` yêu cầu tâm vật nằm trong vùng tương tác của người; vật ở gần chân hoặc ngoài vùng quan tâm bị bỏ.

#### Xác nhận theo thời gian và không gian

Với smartphone, earphone và smartwatch:

- Cửa sổ mặc định: 5 lần object inference.
- Số lần cần xác nhận: 3.
- Detection không chỉ cần cùng class mà còn phải ở cùng vùng vật lý.
- Hai bbox được xem là cùng vật nếu đủ overlap hoặc khoảng cách tâm đủ gần so với kích thước bbox.
- Nhiều false positive ở các vị trí xa nhau không được cộng thành một cảnh báo.
- Điện thoại đang cầm có thể di chuyển nhẹ giữa các inference mà vẫn được nối lại.

Khi class vừa được xác nhận, `_capture_evidence()`:

- Vẽ bbox đỏ lên snapshot.
- Lưu JPG vào `data/sessions/<session_id>/snapshots`.
- Ghi một dòng JSON vào `data/sessions/<session_id>/object_detect_log.jsonl`.
- Không ghi lặp lại mỗi frame khi class vẫn đang ở trạng thái confirmed.

#### Kết quả của `ObjectDetectModule`

Các trường quan trọng:

- `label`: `clear` hoặc dạng `smartphone_detected`.
- `risk_score`: điểm rủi ro tổng hợp.
- `confirmed_classes`: các vật cấm đã đủ xác nhận.
- `raw_detections`: confidence cao nhất theo class.
- `raw_boxes`: bbox tốt nhất theo class.
- `raw_objects`: toàn bộ detection đã qua lọc.
- `paper_detections`: toàn bộ paper candidate chuyển downstream.
- `model_capabilities`: model có class `test_paper` hay không và policy paper đang dùng.
- `inference_ran`: frame này có thật sự chạy YOLO hay chỉ dùng cache.

## 4. Person memory trong `PaperCountPipeline`

File: `backend/ai_services/object_detect/paper_count_pipeline.py`

`PaperCountPipeline` ghép person tracking, object detection và paper-count.

### `create_session(session_id)`

- Tạo session trong `TrackingManager`.
- Tạo session paper-count tương ứng.
- Có tùy chọn restore dữ liệu tracking người đã lưu.

### `process_frame(...)`

Đây là ranh giới an toàn realtime:

- Toàn bộ xử lý được bọc `try...except`.
- Frame lỗi trả `None` thay vì làm sập video/server.
- Có thể giảm tần suất person detection và dùng cache ở frame xen kẽ.

Trong lần person inference thật:

1. `person_detector.detect(frame)` tìm bbox người.
2. `person_fingerprint_from_frame()` trích appearance fingerprint từ vùng người.
3. `TrackingManager.process_detections()` ghép người với track hiện tại hoặc identity đã lưu.
4. Kết quả có `track_id`, `person_id`, `is_present` và trạng thái appearance memory.

`track_id` là ID tạm của tracker. `person_id` là ID do hệ thống/người dùng gán và được appearance memory dùng để nhận lại người khi họ rời camera rồi quay lại.

Sau đó pipeline chuyển bbox người thành `person_rois`, gọi `ObjectDetectModule`, rồi gọi paper monitor nếu frame đó thật sự chạy object inference.

### Kết quả pipeline

```text
session_id
frame_id
timestamp_ms
people
papers
paper_count_state
paper_monitoring_armed
object_result
alerts
risk_score
```

`alerts` là hợp nhất của:

- Cảnh báo paper-count.
- Cảnh báo trực tiếp từ smartphone/earphone/smartwatch.

## 5. Thuật toán `CountBasedPaperMonitor`

File: `backend/ai_services/object_detect/paper_count_monitor.py`

### Nguyên tắc

Monitor không tạo `paper_id`, không lưu appearance fingerprint của giấy và không cố nhận lại từng tờ qua thời gian dài.

Nó chỉ giữ:

- `baseline_count`: số giấy hợp lệ trong SETUP.
- `stable_count`: số giấy đã được xác nhận ổn định.
- `candidate_count` và `candidate_streak`: count đang chờ xác nhận.
- Snapshot bbox baseline dùng để xác định bbox nào là vùng giấy mới.

### Giai đoạn SETUP

Khi chưa `arm`:

1. Mỗi object-inference gom bbox thành paper observation.
2. Lưu count và snapshot gần nhất.
3. Dùng mode, tức count xuất hiện nhiều nhất, làm baseline.
4. Giữ tối đa 60 mẫu setup.

Dùng mode giúp một vài frame YOLO hụt hoặc đếm dư không làm hỏng baseline.

### `cluster_papers(...)`

Một tờ giấy có thể được phát hiện nhiều lần từ:

- Full-frame model chính.
- Custom paper person ROI.
- COCO book fallback.
- Các tile chồng lấn.

Hàm dùng union-find để gom bbox trùng hoặc có tâm đủ gần thành một cluster vật lý. Cluster mới lấy:

- Hộp bao ngoài của cả nhóm.
- Confidence cao nhất.
- Tập class name nguồn.
- Số bbox nguồn `source_box_count`.

Đây là bước chống một tờ bị đếm thành 2–3 tờ.

### `arm(session_id)`

- Khóa baseline từ SETUP.
- Chọn snapshot gần nhất có đúng baseline count.
- Bắt đầu giám sát count tăng.
- Xóa candidate và alert cũ.

### So khớp baseline và frame hiện tại

`_unmatched_current()` ghép một-một bbox baseline với bbox hiện tại dựa trên:

- Intersection over smaller box.
- Khoảng cách tâm đã chuẩn hóa theo đường chéo bbox.

Cluster hiện tại không ghép được với baseline được xem là ứng viên giấy mới. Đây chỉ là so khớp ngắn hạn với snapshot baseline, không phải paper tracking và không tạo ID.

### Điều kiện tạo cảnh báo

Sau khi ARMED, monitor kiểm tra cả:

- Tổng số giấy toàn frame có vượt baseline không.
- Số giấy gần từng người có vượt count baseline của người đó không.

Count tăng phải lặp đủ `paper_count_confirm_inferences`. Giá trị thực tế hiện tại trong `backend/core/config.py` là **2 object-inference**, không phải 2 camera frame.

Khi đủ xác nhận:

- `stable_count` tăng.
- Cluster mới có trạng thái `suspicious_new_paper`.
- Alert có nhãn `cheat_sheet`, nguồn `paper_count`, risk score 1.0.
- Tạo event `paper_count_increased`.

Khi count quay về baseline đủ số inference:

- Xóa active paper alert.
- Tạo event `paper_count_decreased`.

### Gán giấy mới cho người

`_nearest_person()` lấy tâm bbox giấy và chọn người gần nhất:

- Nếu tâm giấy nằm trong bbox người, người đó được ưu tiên rất mạnh.
- Nếu không, chọn bbox người có khoảng cách biên và khoảng cách tâm nhỏ nhất.
- Kết quả gắn cả `owner_track_id` và `owner_person_id`.

### `observation_index` không phải `paper_id`

`observation_index` chỉ là số thứ tự trái sang phải trong frame hiện tại. Nó có thể đổi ở frame sau và không được dùng làm identity.

### Trạng thái trả về

- `mode`: luôn là `count_only`.
- `monitoring_armed`.
- `baseline_count`.
- `observed_count`.
- `stable_count`.
- `candidate_count`.
- `candidate_streak`.
- `confirmation_frames`.
- `papers`.
- `active_alerts`.
- `new_events`.

## 6. Cấu hình quan trọng

File: `backend/core/config.py`

| Cấu hình | Giá trị hiện tại | Ý nghĩa |
|---|---:|---|
| `yolo_model_path` | `weights/best (1).pt` | Model domain chính |
| `object_inference_size` | 640 | Kích thước inference toàn frame |
| `object_detect_every_n_frames` | 2 | Chạy object YOLO mỗi 2 frame xử lý |
| `yolo_confidence_threshold` | 0.50 | Threshold mặc định |
| `earphone` threshold | 0.55 | Threshold riêng cho tai nghe |
| `smartphone` threshold | 0.55 | Threshold riêng của model chính |
| `paper_detection_confidence_threshold` | 0.20 | Threshold giấy toàn frame |
| `object_confirm_frames` | 3 | Số inference xác nhận vật cấm |
| `object_confirm_window` | 5 | Cửa sổ inference cho xác nhận |
| `smartphone_fallback_confidence_threshold` | 0.35 | Threshold phone từ YOLOv8n |
| `book_fallback_confidence_threshold` | 0.15 | Threshold book toàn frame |
| `person_roi_object_inference_size` | 960 | Độ phân giải model phụ trong ROI |
| `person_roi_phone_confidence_threshold` | 0.20 | Ngưỡng phone trong ROI người; vẫn cần xác nhận temporal/spatial |
| `person_roi_custom_paper_inference_size` | 768 | Độ phân giải model giấy trong ROI |
| `person_roi_book_confidence_threshold` | 0.10 | Threshold book trong ROI |
| `paper_count_confirm_inferences` | 2 | Số object inference để xác nhận count |
| `paper_count_duplicate_overlap_threshold` | 0.35 | Ngưỡng gom bbox giấy trùng |
| `paper_count_duplicate_center_distance_ratio` | 0.70 | Ngưỡng tâm để gom bbox |
| `person_appearance_match_threshold` | 0.78 | Ngưỡng nhận lại người bằng appearance |

## 7. Các chương trình test

### `object_detect_test.py`

Test hợp đồng server-facing `ObjectDetector` trên webcam hoặc video. Hiển thị bbox các class cấm sau resize, lọc và quy đổi tọa độ.

### `test_webcam.py`

Test raw `ObjectDetectModule`:

- Cyan: paper candidate chuyển downstream.
- Đỏ: vật cấm không phải giấy đã confirmed.
- Vàng: detection chưa đủ xác nhận.

Test này không chạy person memory hoặc paper-count.

### `test_video_paper_count.py`

Đây là test chính của bản `develop-paper-count`.

- Nhận một hoặc nhiều video.
- Trong SETUP, tự gán người bên trái/phải thành `STUDENT_LEFT` và `STUDENT_RIGHT`.
- Sau `--setup-seconds`, arm baseline giấy.
- Chạy person memory, object detection và count-only monitor.
- Xuất video có bbox và báo cáo JSON.

Lệnh ví dụ:

```powershell
python -m backend.ai_services.object_detect.test_video_paper_count `
  "data/smartphone.mp4" "data/cheatsheet.mp4" `
  --setup-seconds 3 --frame-stride 3
```

Video kết quả:

```text
data/paper_count_results/outputs/<video>_paper_count_output.mp4
```

Báo cáo:

```text
data/paper_count_results/outputs/<video>_paper_count_report.json
```

Báo cáo chứa timeline count, event tăng/giảm, thống kê class, confidence, số inference, trạng thái person memory và đường dẫn video output.

### `test_video_scenarios.py`

Đây là test pipeline cũ có paper identity/tracking. Nó được giữ lại để so sánh với bản count-only, nhưng không phải logic quyết định chính của `develop-paper-count`.

### `test_paper_tracking_webcam.py`

Test webcam end-to-end của pipeline có paper tracking/assignment. Nó cũng được giữ lại để kiểm tra chức năng cũ. Bản count-only hiện có test video riêng nhưng chưa có script webcam riêng tương đương.

### Unit test

`tests/test_object_detect.py` kiểm tra:

- Hợp đồng output và scaling bbox.
- Frame/model lỗi trả `None`.
- Cadence và cache.
- Lọc calculator, remote và bút.
- Xác nhận temporal + spatial.
- Chuẩn hóa cell phone/book.
- NMS theo class.
- Person ROI và owner hint.
- Giữ toàn bộ paper candidate cho downstream.

`tests/test_paper_count_monitor.py` kiểm tra:

- Gom bbox trùng.
- Baseline dùng mode.
- Spike một inference không cảnh báo.
- Count tăng ổn định tạo alert và gán đúng người.
- Quay về baseline xóa alert.
- Tăng giấy theo từng người vẫn được bắt khi tổng count toàn frame không đổi.

## 8. Công cụ chuẩn bị và train paper model

### `prepare_paper_segmentation_frames.py`

- Trích frame từ video để annotate instance segmentation.
- Có tốc độ lấy mẫu thường và tốc độ cao cho hard interval.
- Mỗi tờ giấy vật lý phải được annotate bằng polygon riêng.
- Xuất `manifest.json` chứa video nguồn, frame ID, timestamp và cờ hard interval.

### `notebooks/train_paper_count_yolo26s_seg_kaggle.ipynb`

Notebook train model paper instance segmentation. Mục tiêu là phân tách hai tờ chồng/lệch nhau tốt hơn bbox detection thông thường, từ đó paper-count chính xác hơn.

## 9. Các giới hạn cần nêu rõ

1. Nếu thay tờ cũ bằng tờ mới nhưng tổng count không đổi và vị trí gần giống, count-only có thể không phát hiện.
2. Nếu hai tờ che nhau hoàn toàn và model chỉ trả một region, monitor không thể suy ra có hai tờ.
3. Baseline sai trong SETUP sẽ làm quyết định phía sau sai; camera phải thấy rõ bàn thi trước khi arm.
4. Person owner phụ thuộc bbox người và vị trí tâm giấy; giấy ở giữa hai người có thể gán nhầm.
5. Việc nhận ra giấy nằm trên đùi vẫn phụ thuộc model tạo được bbox/mask. Logic count không thể đếm vật model hoàn toàn bỏ sót.
6. `ObjectDetectModule.process()` không tự bọc `try...except`; ranh giới an toàn được cung cấp bởi `PaperCountPipeline.process_frame()` hoặc `ObjectDetector.process_frame()`.
7. Threshold thấp giúp tăng recall cho paper/book nhưng có thể tăng bbox giả; clustering và xác nhận nhiều inference chỉ giảm chứ không xóa hoàn toàn lỗi model.

## 10. Tóm tắt cơ chế quyết định

- `smartphone`, `earphone`, `smartwatch`: model phát hiện -> lọc hình dạng/confuser -> khớp cùng vị trí qua nhiều inference -> alert.
- `book`, `cheat_sheet`, `test_paper`, `paper_unknown`: model phát hiện -> chuẩn hóa thành paper candidate -> gom bbox trùng -> đếm -> so baseline -> alert nếu count tăng ổn định.
- Người: person detector -> appearance fingerprint -> tracking manager -> `person_id` ổn định -> paper mới được gán cho người gần nhất.
- Giấy: không có `paper_id`; `observation_index` chỉ dùng để vẽ trong frame.
