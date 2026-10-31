"""Pipeline dùng chung cho luồng realtime (app_webrtc*.py) và luồng offline (run_offline.py).

Mỗi module ở đây tương ứng với một đoạn logic từng bị nhốt trong closure của
`init_system_resources()` trong app_webrtc.py. Quy ước chung:

- Mọi hàm đều **thuần về phía AI**: nhận frame/kết quả, trả về `alert dict` hoặc `None`.
- Không giữ state toàn cục; state (queue, overlay list, session) thuộc về caller.
- `alert dict` luôn có khoá `module`, `status`, `timestamp` — xem `events.to_log_line()`.
"""
