import os
import numpy as np
import librosa
from scipy.signal import butter, lfilter
from sklearn.metrics import f1_score, classification_report

# Import class tổng của hệ thống
from backend.ai_services.whisper.audio_pipeline import AudioPipeline

TARGET_SR = 16000

# ==========================================
# CÁC HÀM XỬ LÝ ÂM THANH THỰC TẾ
# ==========================================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=300.0, highcut=3400.0, fs=16000, order=4):
    """Giữ lại dải tần giọng nói, triệt tiêu tiếng ù quạt và tiếng chói"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y.astype(np.float32)

def load_wav_and_preprocess(filepath):
    """Hàm load file wav và tiền xử lý"""
    # Load audio gốc bằng librosa (chuyển về mono, 16000Hz)
    audio, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
    
    # ==========================================
    # 🔴 ĐÂY LÀ MỐC 2: BẬT LỌC BANDPASS (300Hz - 3400Hz)
    # ==========================================
    audio = apply_bandpass_filter(audio) 
    
    return audio

# ==========================================
# HÀM ĐÁNH GIÁ (EVALUATE PIPELINE)
# ==========================================
def evaluate_folder():
    # Tắt các cảnh báo Hugging Face để log sạch sẽ
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    
    print("[System] Đang khởi tạo hệ thống nhận diện...")
    pipeline = AudioPipeline()
    y_true = []
    y_pred = []
    
    # 1. Quét file Bình thường (Nhãn 0: Normal)
    normal_dir = "backend/ai_services/whisper/samples/normal"
    if os.path.exists(normal_dir):
        for file in os.listdir(normal_dir):
            if file.endswith(".wav"):
                filepath = os.path.join(normal_dir, file)
                clean_audio = load_wav_and_preprocess(filepath)
                
                # Chạy qua luồng AI
                result = pipeline.process_audio(clean_audio)
                
                y_true.append(0)
                # Dựa vào status 'alert' để xác định hệ thống báo động hay không
                is_alert = 1 if result and result.get('status', '').lower() == 'alert' else 0
                y_pred.append(is_alert)
                
                if is_alert == 1:
                    print(f"❌ Bắt nhầm (False Positive): {file} | Text: {result.get('transcription')}")
    else:
        print(f"⚠️ Không tìm thấy thư mục {normal_dir}. Vui lòng tạo thư mục và thêm file .wav")

    # 2. Quét file Gian lận (Nhãn 1: Cheating)
    cheating_dir = "backend/ai_services/whisper/samples/cheating"
    if os.path.exists(cheating_dir):
        for file in os.listdir(cheating_dir):
            if file.endswith(".wav"):
                filepath = os.path.join(cheating_dir, file)
                clean_audio = load_wav_and_preprocess(filepath)
                
                result = pipeline.process_audio(clean_audio)
                
                y_true.append(1)
                is_alert = 1 if result and result.get('status', '').lower() == 'alert' else 0
                y_pred.append(is_alert)
                
                if is_alert == 0:
                    print(f"⚠️ Bỏ lọt (False Negative): {file} | Text: {result.get('transcription')}")
    else:
        print(f"⚠️ Không tìm thấy thư mục {cheating_dir}. Vui lòng tạo thư mục và thêm file .wav")

    # 3. Tính toán và in Bảng điểm F1
    if len(y_true) > 0:
        print("\n" + "="*55)
        print("KẾT QUẢ MỐC 2: CÓ BANDPASS FILTER (300-3400Hz)")
        print("="*55)
        print(classification_report(y_true, y_pred, target_names=["Normal", "Cheating"]))
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"🔥 Macro F1-Score: {f1:.4f}")
        print("="*55)

if __name__ == "__main__":
    evaluate_folder()