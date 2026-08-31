from ultralytics import YOLO

print("[INFO] Đang nạp file mô hình PyTorch của cộng sự...")
# Trỏ đúng đến file .pt hiện tại của bạn
model = YOLO("weights/best (1).pt")

print("[INFO] Đang tiến hành chuyển đổi sang định dạng OpenVINO...")
# Lệnh này sẽ tự động sinh ra thư mục chứa file .xml và .bin
model.export(format="openvino")

print("[INFO] ✅ Export thành công! Thư mục OpenVINO đã được tạo.")