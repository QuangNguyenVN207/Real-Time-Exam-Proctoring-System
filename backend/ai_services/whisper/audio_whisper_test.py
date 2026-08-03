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
import warnings
import transformers

# Tắt các cảnh báo cơ bản của Python (nếu có)
warnings.filterwarnings("ignore")

# Ép thư viện transformers chỉ in ra lỗi (Error), ẩn đi các Cảnh báo (Warning)
transformers.logging.set_verbosity_error()

class AudioWhisper:
    def __init__(self, phowhisper_model="vinai/PhoWhisper-tiny", phobert_model_path="data/phobert_gian_lan_final"):
        print("[INFO] Đang khởi tạo VAD, PhoWhisper và PhoBERT...")
        
        # 1. Khởi tạo VAD (Mức 3 - Khắt khe nhất)
        self.vad = webrtcvad.Vad(1)  # Mức 1 là ít khắt khe, 3 là khắt khe nhất
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
        
        # # --- BƯỚC 4: PHOBERT (Đọc hiểu ý định) ---
        # if self.phobert_model is None:
        #     return {"lỗi": "Chưa có mô hình PhoBERT để phân loại"}

        # inputs = self.phobert_tokenizer(segmented_text, return_tensors="pt", truncation=True, max_length=256)
        
        # with torch.no_grad():
        #     outputs = self.phobert_model(**inputs)
            
        # # Lấy nhãn có xác suất cao nhất
        # logits = outputs.logits
        # predicted_class_id = logits.argmax().item()
        
        # # Giả sử trong tập train của bạn: 0 là Hợp_lệ, 1 là Gian_lận
        # is_cheating = (predicted_class_id == 1)
        
        # # --- BƯỚC 5: TRẢ KẾT QUẢ ---
        # if is_cheating:
        #     return {
        #         "module": "audio_phobert_pipeline",
        #         "status": "alert",
        #         "timestamp": float(timestamp),
        #         "transcription": raw_text,         # Lưu lại câu gốc để giám thị đọc cho dễ
        #         "segmented_text": segmented_text,  # Dạng đã xử lý của PhoBERT
        #         "intent": "Gian_lận",
        #         "ai_confidence": round(torch.softmax(logits, dim=1)[0][1].item(), 2) # Độ tự tin của AI
        #     }
            
        # return None

        # --- BƯỚC 4: PHOBERT (Đọc hiểu ý định) ---
        if self.phobert_model is None:
            return {"lỗi": "Chưa có mô hình PhoBERT để phân loại"}

        inputs = self.phobert_tokenizer(segmented_text, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = self.phobert_model(**inputs)
            
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        
        is_cheating = (predicted_class_id == 1)
        # Lấy độ tự tin của nhãn vừa được dự đoán
        confidence = round(torch.softmax(logits, dim=1)[0][predicted_class_id].item(), 2)
        
        # --- BƯỚC 5: TRẢ KẾT QUẢ CHO MỌI TRƯỜNG HỢP ---
        # Thay vì chỉ trả về khi gian lận, giờ ta trả về mọi lúc
        return {
            "module": "audio_phobert_pipeline",
            "status": "alert" if is_cheating else "normal",
            "timestamp": float(timestamp),
            "transcription": raw_text,         
            "segmented_text": segmented_text,  
            "intent": "Gian_lận" if is_cheating else "Hợp_lệ",
            "ai_confidence": confidence
        }

# Chạy thử nghiệm với Microphone thực tế
if __name__ == "__main__":
    import time
    import sounddevice as sd
    import json
    import numpy as np
    import scipy.io.wavfile as wavfile 
    import scipy.signal 
    
    # 1. KHẮC PHỤC LỖI ĐƯỜNG DẪN PHOBERT
    # LƯU Ý QUAN TRỌNG: Bạn hãy đảm bảo thư mục 'phobert_gian_lan_final' 
    # đang nằm CÙNG CHỖ với file script bạn đang chạy.
    detector = AudioWhisper(
        phowhisper_model="vinai/PhoWhisper-tiny", 
        phobert_model_path="data/phobert_gian_lan_final" 
    )
    
    DURATION = 5  
    TARGET_RATE = 16000 
    RECORD_RATE = 48000 
    
    print("\n" + "="*50)
    print(f"🎤 BẮT ĐẦU TEST: Hệ thống sẽ ghi âm {DURATION} giây.")
    print("="*50 + "\n")
    
    # 2. BẬT MICRO (Gọi đích danh thiết bị 3, tần số 48k, và thêm blocking=True để chống treo)
    print("[HỆ THỐNG] Đang thu âm... (Hãy nói to và rõ!)")
    
    # Thêm device=3 và blocking=True
    audio_data = sd.rec(
        int(DURATION * RECORD_RATE), 
        samplerate=RECORD_RATE, 
        channels=1, 
        dtype='float32', 
        device=3,           
        blocking=True       
    )
    # sd.wait() 
    
    print("[HỆ THỐNG] Đã thu âm xong.")
    
    audio_chunk_48k = audio_data.flatten()
    num_samples_16k = int(len(audio_chunk_48k) * (TARGET_RATE / RECORD_RATE))
    audio_chunk = scipy.signal.resample(audio_chunk_48k, num_samples_16k)
    
    max_amp = np.max(np.abs(audio_chunk))
    print(f"📊 Mức âm lượng cao nhất thu được: {max_amp:.5f}")
    
    # Ép kiểu về int16 để đảm bảo file wav phát được trên mọi trình nghe nhạc của Ubuntu
    audio_chunk_int16 = (audio_chunk * 32767).astype(np.int16)
    wavfile.write("kiem_tra_micro.wav", TARGET_RATE, audio_chunk_int16)
    print("💾 Đã lưu file 'kiem_tra_micro.wav'.")
    
    timestamp_now = time.time()
    
    print("\n[HỆ THỐNG] Đang xử lý AI...")
    result = detector.process_audio(audio_chunk, timestamp_now)
    
    # 3. SỬA LỖI KEYERROR: Kiểm tra an toàn trước khi in
    print("\n" + "="*20 + " KẾT QUẢ " + "="*20)
    if result:
        # Nếu AI trả về thông báo lỗi thiếu mô hình
        if "lỗi" in result:
            print(f"\033[93m[LỖI HỆ THỐNG] {result['lỗi']}\033[0m")
            print("Vui lòng giải nén file Colab và đặt đúng thư mục!")
        else:
            print(f"🗣️ BẠN ĐÃ NÓI: '{result['transcription']}'")
            print(f"🤖 VĂN BẢN XỬ LÝ: '{result['segmented_text']}'\n")
            
            if result["status"] == "alert":
                print(f"\033[91m🚨 PHÁT HIỆN GIAN LẬN! (Độ tự tin: {result['ai_confidence']*100}%)\033[0m")
            else:
                print(f"\033[92m✅ HỢP LỆ. (Độ tự tin: {result['ai_confidence']*100}%)\033[0m")
    else:
        print("Trống. (VAD không phát hiện tiếng người, hoặc âm thanh quá ồn).")
    print("="*49 + "\n")