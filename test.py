from pathlib import Path

# Đường dẫn tới file calibration.json của bạn
calib_path = Path("tmp/causal_8fps_stage6_mixed_084699_final_20260827/calibration.json")

if calib_path.exists():
    # Đọc nội dung file
    content = calib_path.read_text(encoding="utf-8")
    # Ghi lại với chuẩn xuống dòng LF ('\n')
    calib_path.write_text(content, encoding="utf-8", newline="\n")
    print("✅ Đã chuẩn hóa định dạng file calibration.json sang LF thành công!")
else:
    print(f"❌ Không tìm thấy file tại: {calib_path}")