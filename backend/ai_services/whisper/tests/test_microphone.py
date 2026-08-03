from __future__ import annotations
import sys
from pathlib import Path
import time
import numpy as np
import scipy.signal
import sounddevice as sd

# Trỏ đường dẫn để Python nhận diện thư mục ai_services
AI_SERVICES_DIR = Path(__file__).resolve().parent.parent.parent
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

# Import thẳng bộ não 87% Accuracy của chúng ta!
from whisper.audio_pipeline import AudioPipeline

def get_default_mic_channels(device_id=None):
    """Tự động tìm số kênh của Microphone đang dùng"""
    device_info = sd.query_devices(device_id, 'input')
    return device_info['max_input_channels']

def main():
    print("="*60)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG GIÁM THỊ ÂM THANH THỰC TẾ (REAL-TIME)")
    print("="*60)
    
    # 1. Khởi tạo toàn bộ hệ thống (VAD, Whisper, Keyword, PhoBERT, Fusion, Logger)
    pipeline = AudioPipeline()
    print("\n[✓] Hệ thống AI đã sẵn sàng đón nhận âm thanh!")
    
    # 2. Cấu hình thu âm
    DURATION = 6          # Thu âm mỗi nhịp 6 giây
    RECORD_RATE = 48000   # Tần số thực của mic
    TARGET_RATE = 16000   # Tần số bắt buộc của AI
    DEVICE_ID = None      # Để None để tự động lấy Mic mặc định (Hoặc thay bằng số 3 như code cũ)
    
    channels = get_default_mic_channels(DEVICE_ID)
    
    input("\n🎤 Bấm [ENTER] để bắt đầu thu âm ngay lập tức...")
    
    print(f"\n🔴 ĐANG THU ÂM TRONG {DURATION} GIÂY... (Hãy nói gì đó!)")
    
    # 3. Tiến hành thu âm bằng SoundDevice
    audio_data = sd.rec(
        int(DURATION * RECORD_RATE), 
        samplerate=RECORD_RATE, 
        channels=channels, 
        dtype='float32', 
        device=DEVICE_ID,
        blocking=True # Đợi thu xong mới chạy tiếp
    )
    
    print("⏹️ Đã thu âm xong. Đang đưa vào AI phân tích...\n")
    
    # ================================================================
    # 4. TIỀN XỬ LÝ & KHUẾCH ĐẠI ÂM LƯỢNG (Khắc phục lỗi mic nhỏ)
    # ================================================================
    
    # A. Xử lý lỗi mic Stereo (Nếu mic 2 kênh, phải gộp thành Mono)
    if channels > 1:
        audio_mono = np.mean(audio_data, axis=1) # Lấy trung bình L và R
    else:
        audio_mono = audio_data.flatten()
        
    # B. KHUẾCH ĐẠI ÂM LƯỢNG (SOFTWARE GAIN)
    # 🔊 Tăng con số này lên nếu mic vẫn nhỏ (thử 2.0, 3.0, hoặc 5.0)
    VOLUME_MULTIPLIER = 3.0  
    audio_boosted = audio_mono * VOLUME_MULTIPLIER
    
    # Cắt cúp (clip) những đoạn sóng âm vượt quá mức cho phép để không bị rè/chói tai
    audio_boosted = np.clip(audio_boosted, -1.0, 1.0)
    
    # C. Ép tần số từ mic (thường 48k) xuống 16k cho AI hiểu
    num_samples_16k = int(len(audio_boosted) * (TARGET_RATE / RECORD_RATE))
    audio_chunk_16k = scipy.signal.resample(audio_boosted, num_samples_16k).astype(np.float32)
    
    # ================================================================
    # 5. ĐƯA VÀO ĐƯỜNG ỐNG PIPELINE CỦA CHÚNG TA
    # ================================================================
    start_time = time.time()
    result = pipeline.process_audio(audio_chunk_16k, source="microphone")
    end_time = time.time()
    
    # 6. IN KẾT QUẢ ĐẸP MẮT
    print("="*60)
    if result is None:
        print("Trống. (VAD không phát hiện tiếng người, hoặc bạn nói quá nhỏ).")
    else:
        print(f"⏱️ THỜI GIAN DỊCH: {end_time - start_time:.3f} giây")
        
        # Đánh giá cảnh báo
        print("\n" + "=" * 70)
        
        # Gom tất cả các nhãn có thể là gian lận ('cheating', 'alert', 'high', 'medium')
        risk_final = str(result.get('risk', '')).lower()
        status_final = str(result.get('status', '')).lower()
        
        if risk_final in ['cheating', 'high', 'medium'] or status_final == 'alert':
            # Dựa vào lý do để biết chính xác là Keyword hay AI bắt được
            reason_text = str(result.get('fusion_reason', '')).lower()
            if 'ai catch' in reason_text:
                nguoi_bat = "🤖 AI PhoBERT (Bọc lót Keyword)"
            else:
                nguoi_bat = "🔑 Keyword (Bộ luật cứng)"
                
            print(f"🚨 [CẢNH BÁO GIAN LẬN] - Phát hiện bởi: {nguoi_bat}")
        else:
            print(f"✅ [AN TOÀN] - Cả Keyword và AI đều đồng ý an toàn")
            
        print(f"🗣️ Transcript : '{result.get('transcription', '')}'")
        print(f"🧠 Lý do      : {result.get('fusion_reason', '')}")
        
        keywords = result.get('matched_keywords', [])
        if keywords:
            print("🔑 Từ khóa bị bắt:")
            for kw in keywords:
                print(f"   - '{kw['keyword']}' (Mức độ: {kw['severity']})")

    print("="*60)
    print("Mọi lịch sử vừa rồi đã được ghi tự động vào file audio_log.jsonl!")

if __name__ == "__main__":
    main()