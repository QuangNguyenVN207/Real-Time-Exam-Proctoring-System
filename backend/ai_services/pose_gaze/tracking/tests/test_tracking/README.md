# Test tracking

## Ý tưởng

Bảo vệ các invariant của IoU tracking, assignment và session persistence khi
cấu trúc package hoặc lifecycle thay đổi.

## Input

Detection/bbox giả lập, temporary storage và detector fake; không cần webcam,
YOLO hoặc MediaPipe.

## Output

Unit-test assertions cho ID ổn định, outsider, grace period, restore, legacy
state, assignment, module frame counter và FPS controller.
