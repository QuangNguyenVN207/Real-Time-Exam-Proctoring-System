# Generated CSV data

Folder này là output mặc định của batch runner:

- `train.csv`
- `val.csv`
- `test.csv`

Ba CSV và các file `.part` có thể rất lớn nên bị Git ignore; chỉ README này được
commit. Chạy batch với `--overwrite` mới được phép thay kết quả đã có.

Mỗi hàng tương ứng một người trong một ảnh/frame. Dùng `status=ok` cho train;
giữ các status khác để audit tỷ lệ ảnh lỗi, không thấy người hoặc thiếu
landmark.
