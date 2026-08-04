# Batch dataset

## Ý tưởng

Đọc ảnh tuần tự, chỉ giữ một ảnh trong RAM, rồi chạy:

```text
image -> YOLO person -> IoU track trong frame -> Holistic -> fixed CSV row
```

Detector và MediaPipe model được khởi tạo đúng một lần để phù hợp với tập hàng
nghìn ảnh. Mỗi người được ghi thành một hàng; ảnh không đọc được, không thấy
người hoặc không có landmark vẫn có hàng với `status` tương ứng.

## Input

Cách 1 — dữ liệu đã chia split:

```text
dataset/
  train/c2__hand_reach_toward_friend/*.jpg
  train/c2__hand_reach_toward_friend/video_001/frame_*.jpg
  val/c2__hand_reach_toward_friend/*.jpg
  test/c5__normal/*.jpg
```

Cách 2 — chưa chia split:

```text
dataset/
  c2__hand_reach_toward_friend/*.jpg
  c5__normal/*.jpg
```

Runner chia ổn định 80/10/10 bằng hash. Nếu có folder `sequence_id`, toàn bộ
frame trong sequence luôn vào cùng split; ảnh độc lập mới hash theo đường dẫn
ảnh. Đổi bằng `--train-ratio`, `--val-ratio`, `--seed`. Dataset thật nên ưu tiên
manifest có `split_group`/`split` để tránh subject hoặc session leakage.

Cách 3 — `--manifest annotations.tsv`: hỗ trợ các cột note đã cung cấp như
`filename`, `class_code`, `label`, `split`, `split_group`, actor, action actor,
`observed_*`, `exclude_from_training`, quality và note. Header được đọc không
phân biệt hoa thường. Manifest này phải trỏ tới **ảnh**; video cần được tách
frame trước.

File `manifest.example.tsv` là template đã chuẩn hóa từ note để tạo manifest
frame-level.

Ảnh đặt trực tiếp dưới folder class được coi là độc lập và tracker reset cho
từng ảnh. Nếu ảnh là frame liên tiếp của video, đặt chúng trong folder
`<class>/<sequence_id>/` hoặc thêm cột `sequence_id` vào manifest; tracker khi
đó được giữ xuyên các frame của đúng sequence.

## Output

Mặc định ghi cạnh file chạy tại `data/train.csv`, `data/val.csv`,
`data/test.csv`. File được stream qua `.part` và chỉ publish sau khi toàn bộ run
thành công. Không dùng `--overwrite` nếu muốn bảo vệ kết quả cũ.

Mỗi hàng chứa metadata annotation/tracking và feature landmark cố định. Khi
train, lọc ít nhất:

```text
status == "ok" and exclude_from_training != "TRUE"
```

Không tự gán `action_actor_ids` vào `track_id`; với frame có nhiều người cần
mapping actor-track hoặc crop riêng actor trước khi train.

## Chạy

```powershell
python -m backend.ai_services.pose_gaze.holistic.batch_dataset D:\dataset `
  --model .\weights\yolov8n.pt `
  --device 0
```

Chạy thử 100 ảnh và ghi đè output cũ:

```powershell
python -m backend.ai_services.pose_gaze.holistic.batch_dataset D:\dataset `
  --limit 100 --overwrite
```

## Công năng và nhiệm vụ

- Discover ảnh và split tái lập được.
- Tái sử dụng detector/Holistic, không giữ toàn bộ dataset trong RAM.
- Ghi đủ ảnh lỗi/không detect bằng `status`.
- Không sinh `z`.
- Giữ `class_code` làm target chính và `label` làm mô tả.
