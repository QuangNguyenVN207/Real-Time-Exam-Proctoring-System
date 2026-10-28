"""Lối tắt khởi động trên Windows — toàn bộ logic nằm trong `app_webrtc.py`.

`app_webrtc.py` là entry point duy nhất và tự nhận nền tảng (`sys.platform`):
Windows/NVIDIA -> `select_device()` + worker micro `sounddevice` (`realtime_audio_worker.py`),
Ubuntu/iGPU   -> OpenVINO + worker micro `PyAudio` (`realtime_audio_ubuntu.py`).

Giữ file này chỉ để lệnh `streamlit run app_webrtc_win.py` (và các lối tắt cũ) không đổi.
"""
import pathlib
import runpy

runpy.run_path(str(pathlib.Path(__file__).with_name("app_webrtc.py")), run_name="__main__")
