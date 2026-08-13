import sounddevice as sd
import numpy as np
import queue
import threading
import time
from datetime import datetime

# 1. IMPORT BỘ NÃO AI CỦA BẠN (Thay cho OpenAI Whisper cũ)
from audio_pipeline import AudioPipeline

print("Đang khởi tạo AI Giám thị (VAD + PhoBERT + PhoWhisper)...")
pipeline = AudioPipeline()
print("Khởi tạo xong! Sẵn sàng.")

# Cấu hình thu âm
samplerate = 16000
channels = 1
# 2. CHỈNH VỀ 5 GIÂY (Dự án gốc để 30 giây là quá chậm)
blocksize = 3.5 * samplerate  
audio_queue = queue.Queue()
recording = True

def audio_callback(indata, frames, time, status):
    """Hứng âm thanh từ Mic"""
    if status:
        pass # Bỏ qua in lỗi lặt vặt của mic
    audio_queue.put(indata.copy())

def process_audio():
    """Luồng xử lý âm thanh"""
    while recording:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            audio_data = audio_data.flatten().astype(np.float32)
            
            try:
                # 3. ĐƯA VÀO ĐƯỜNG ỐNG AI CỦA BẠN XỬ LÝ
                result = pipeline.process_audio(audio_data)
                
                # 4. TRẢ KẾT QUẢ RA CHO GUI HIỂN THỊ
                if result and result.get("status") != "idle":
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    status = result.get("status", "")
                    text = result.get("transcription", "")
                    reason = result.get("fusion_reason", "")
                    
                    # Vì GUI thường dùng sys.stdout để bắt chữ lên màn hình,
                    # ta dùng hàm print để đẩy text thẳng lên giao diện của họ
                    if status == "alert":
                        print(f"🚨 [{timestamp}] GIAN LẬN: {text} - {reason}")
                    else:
                        print(f"✅ [{timestamp}] AN TOÀN: {text}")
                        
            except Exception as e:
                print(f"Lỗi: {str(e)}")

def start_recording():
    """Hàm này để whisper_gui.py gọi khi bấm nút Bắt đầu"""
    global recording
    recording = True
    process_thread = threading.Thread(target=process_audio, daemon=True)
    process_thread.start()
    
    # Khởi động stream (Chạy ngầm không block)
    stream = sd.InputStream(samplerate=samplerate,
                            channels=channels,
                            callback=audio_callback,
                            blocksize=blocksize)
    stream.start()
    return stream

def stop_recording(stream):
    """Hàm này để whisper_gui.py gọi khi bấm nút Dừng"""
    global recording
    recording = False
    if stream:
        stream.stop()
        stream.close()