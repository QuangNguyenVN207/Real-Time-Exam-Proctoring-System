"""Lối tắt khởi động trên Windows — toàn bộ logic nằm trong `app_webrtc.py`.

`app_webrtc.py` là entry point duy nhất. Model + thiết bị suy luận chọn theo **kind của thiết bị**
mà `select_device()` resolve được (`cuda` / `intel` / `cpu`, xem §5 README), nên chạy native Windows,
WSL2 hay Linux đều ra đúng cấu hình cho phần cứng — chỉ worker micro là theo OS
(`sounddevice` trên Windows, `PyAudio` trên Linux/WSL).

Giữ file này chỉ để lệnh `streamlit run app_webrtc_win.py` (và các lối tắt cũ) không đổi.
"""
import pathlib
import runpy

runpy.run_path(str(pathlib.Path(__file__).with_name("app_webrtc.py")), run_name="__main__")
