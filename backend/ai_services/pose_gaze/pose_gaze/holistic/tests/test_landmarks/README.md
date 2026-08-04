# Test landmarks

## Ý tưởng

Kiểm tra các helper thuần Python mà không cần khởi tạo MediaPipe model hoặc mở
camera.

## Input

Landmark fake dạng Tasks/legacy, bbox, letterbox transform và temporary JSON
writer.

## Output

Assertions cho filter index, schema chỉ có x/y, confidence chuẩn/mềm/thấp,
optional protobuf score, timestamp, writer v2 và shape letterbox.
