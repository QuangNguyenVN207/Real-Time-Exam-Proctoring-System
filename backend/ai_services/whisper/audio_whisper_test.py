# import random
# import time

# class AudioWhisper:
#     def __init__(self, model_path=None):
#         print("[MOCK] Đã khởi tạo AudioWhisper (VAD + Whisper giả lập)")

#     def process_audio(self, audio_chunk, timestamp):
#         # Tỉ lệ 2% phát hiện tiếng nhắc bài
#         if random.random() < 0.02:
#             return {
#                 "module": "audio_whisper",
#                 "status": "alert",
#                 "timestamp": timestamp,
#                 "details": {
#                     "transcription": "cau 5 dap an la A phai khong",
#                     "keyword_detected": True
#                 }
#             }
#         return None # Im lặng hoặc tiếng ồn môi trường

import numpy as np
import webrtcvad
import string
import time
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyvi import ViTokenizer # Thư viện tách từ tiếng Việt

class AudioWhisper:
    def __init__(self, phowhisper_model="vinai/PhoWhisper-tiny", phobert_model_path="data/phobert_gian_lan_final"):
        print("[INFO] Đang khởi tạo VAD, PhoWhisper và PhoBERT...")
        
        # 1. Khởi tạo VAD (Mức 3 - Khắt khe nhất)
        self.vad = webrtcvad.Vad(3)
        self.sample_rate = 16000
        
        # 2. Khởi tạo PhoWhisper (Speech-to-Text)
        self.whisper_processor = WhisperProcessor.from_pretrained(phowhisper_model)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(phowhisper_model)
        
        # 3. Khởi tạo PhoBERT (Intent Classification - Phân loại gian lận)
        # Lưu ý: Đây phải là model PhoBERT đã được bạn fine-tune bằng dữ liệu phòng thi
        try:
            self.phobert_tokenizer = AutoTokenizer.from_pretrained(phobert_model_path)
            self.phobert_model = AutoModelForSequenceClassification.from_pretrained(phobert_model_path)
        except Exception as e:
            print("[CẢNH BÁO] Chưa tìm thấy mô hình PhoBERT đã train. Vui lòng cập nhật đường dẫn.")
            self.phobert_model = None

        print("[INFO] Hệ thống sẵn sàng!")

    def _is_speech_present(self, audio_chunk_np):
        # Chuyển đổi numpy array sang PCM 16-bit cho WebRTCVAD
        audio_pcm16 = (audio_chunk_np * 32767).astype(np.int16).tobytes()
        frame_duration = 30 # ms
        frame_size = int(self.sample_rate * (frame_duration / 1000.0) * 2)
        
        active_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio_pcm16) - frame_size, frame_size):
            frame = audio_pcm16[i:i + frame_size]
            total_frames += 1
            if self.vad.is_speech(frame, self.sample_rate):
                active_frames += 1
                
        # Ngưỡng 15% khung thời gian có tiếng động
        return total_frames > 0 and (active_frames / total_frames) > 0.15

    def process_audio(self, audio_chunk, timestamp):
        # Đảm bảo đầu vào là float32 numpy array
        if isinstance(audio_chunk, bytes):
            audio_chunk = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32767.0
            
        # --- BƯỚC 1: VAD GÁC CỔNG ---
        if not self._is_speech_present(audio_chunk):
            return None # Bỏ qua nếu chỉ là tiếng ồn
            
        # --- BƯỚC 2: PHOWHISPER (Nghe) ---
        input_features = self.whisper_processor(
            audio_chunk, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features
        
        predicted_ids = self.whisper_model.generate(input_features)
        raw_text = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        
        if not raw_text:
            return None
            
        # --- BƯỚC 3: TIỀN XỬ LÝ & TÁCH TỪ (Dọn dẹp) ---
        # Chuyển chữ thường và xóa dấu câu
        clean_text = raw_text.lower().translate(str.maketrans('', '', string.punctuation))
        # Nối từ ghép bằng dấu gạch dưới (VD: "đáp án" -> "đáp_án")
        segmented_text = ViTokenizer.tokenize(clean_text)
        
        # --- BƯỚC 4: PHOBERT (Đọc hiểu ý định) ---
        if self.phobert_model is None:
            return {"lỗi": "Chưa có mô hình PhoBERT để phân loại"}

        inputs = self.phobert_tokenizer(segmented_text, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = self.phobert_model(**inputs)
            
        # Lấy nhãn có xác suất cao nhất
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        
        # Giả sử trong tập train của bạn: 0 là Hợp_lệ, 1 là Gian_lận
        is_cheating = (predicted_class_id == 1)
        
        # --- BƯỚC 5: TRẢ KẾT QUẢ ---
        if is_cheating:
            return {
                "module": "audio_phobert_pipeline",
                "status": "alert",
                "timestamp": float(timestamp),
                "transcription": raw_text,         # Lưu lại câu gốc để giám thị đọc cho dễ
                "segmented_text": segmented_text,  # Dạng đã xử lý của PhoBERT
                "intent": "Gian_lận",
                "ai_confidence": round(torch.softmax(logits, dim=1)[0][1].item(), 2) # Độ tự tin của AI
            }
            
        return None

# Chạy thử nghiệm với Microphone thực tế
if __name__ == "__main__":
    import time
    import sounddevice as sd
    import json
    
    # 1. Khởi tạo Pipeline
    # LƯU Ý: Sửa đường dẫn './phobert_gian_lan_final' thành đường dẫn đúng tới thư mục bạn vừa giải nén
    detector = AudioWhisper(
        phowhisper_model="vinai/PhoWhisper-tiny", 
        phobert_model_path="data/phobert_gian_lan_final"
    )
    
    DURATION = 5  # Thời gian ghi âm mỗi lần test (5 giây)
    SAMPLE_RATE = 16000 # Chuẩn âm thanh bắt buộc của Whisper và VAD
    
    print("\n" + "="*50)
    print(f"🎤 BẮT ĐẦU TEST: Hệ thống sẽ ghi âm {DURATION} giây.")
    print("Hãy thử nói một câu gian lận (vd: 'đáp án câu năm là gì mày')")
    print("="*50 + "\n")
    
    # 2. Bật Micro và ghi âm
    print("[HỆ THỐNG] Đang thu âm... (Hãy nói đi!)")
    # Thu âm 1 kênh (mono), định dạng float32
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait() # Chờ đủ 5 giây
    
    print("[HỆ THỐNG] Đã thu âm xong. Đang xử lý AI...")
    
    # 3. Chuyển đổi dữ liệu micro thành mảng 1D cho Pipeline
    audio_chunk = audio_data.flatten()
    timestamp_now = time.time()
    
    # 4. Đưa vào đường ống phân tích
    result = detector.process_audio(audio_chunk, timestamp_now)
    
    # 5. In kết quả ra Terminal
    print("\n" + "="*20 + " KẾT QUẢ " + "="*20)
    if result:
        # Nếu có phát hiện gian lận, in ra JSON cảnh báo có màu mè cho dễ nhìn
        print("\033[91m🚨 PHÁT HIỆN DẤU HIỆU GIAN LẬN! 🚨\033[0m")
        print(json.dumps(result, indent=4, ensure_ascii=False))
    else:
        print("\033[92m✅ An toàn.\033[0m (Không có lời nói, hoặc lời nói hợp lệ).")
    print("="*49 + "\n")