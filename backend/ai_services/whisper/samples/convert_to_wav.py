import os
import librosa
import soundfile as sf
from pydub import AudioSegment
# 1. Khai báo thư mục
# Thay đổi đường dẫn này thành nơi bạn đang để đống file .mp3 và .m4a
SOURCE_DIR = "data/test_audio"  # Thư mục chứa các file mp3/m4a gốc

# Thư mục đích mà evaluate_pipeline.py đang đọc
NORMAL_DIR = "backend/ai_services/whisper/samples/normal"
CHEATING_DIR = "backend/ai_services/whisper/samples/cheating"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(CHEATING_DIR, exist_ok=True)

def convert_and_sort():
    # Quét tất cả các file trong thư mục nguồn
    for filename in os.listdir(SOURCE_DIR):
        if filename.lower().endswith((".mp3", ".m4a")):
            source_path = os.path.join(SOURCE_DIR, filename)
            
            # 2. Tự động phân loại dựa vào tên file
            if "normal" in filename.lower():
                dest_dir = NORMAL_DIR
            elif "cheat" in filename.lower():
                dest_dir = CHEATING_DIR
            else:
                print(f"Bỏ qua file không rõ nhãn: {filename}")
                continue
            
            # 3. Tạo tên file mới với đuôi .wav
            new_filename = os.path.splitext(filename)[0] + ".wav"
            dest_path = os.path.join(dest_dir, new_filename)
            
            print(f"Đang xử lý: {filename} -> {dest_dir}/{new_filename}...")
            
            try:
                # Đọc file bằng pydub (tự động nhận diện format)
                audio = AudioSegment.from_file(source_path)
                
                # Ép về Mono (1 channel) và Sample Rate 16000Hz 
                audio = audio.set_channels(1).set_frame_rate(16000)
                
                # Xuất ra file .wav
                audio.export(dest_path, format="wav")
            except Exception as e:
                print(f"❌ Lỗi khi chuyển đổi {filename}: {e}")
                
    print("\n✅ Hoàn tất! Toàn bộ file đã được chuyển thành .wav (16kHz, Mono) và đưa vào đúng thư mục.")

if __name__ == "__main__":
    convert_and_sort()