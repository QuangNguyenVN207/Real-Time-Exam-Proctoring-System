import time
import numpy as np

# Các tiện ích tiền xử lý
from whisper.audio_utils import load_audio, extract_speech, apply_bandpass_filter

# Import các AI Services hiện có
from whisper.vad_service import VADService
from whisper.phowhisper_service import PhoWhisperService
from whisper.keyword_detector import KeywordDetector
from whisper.audio_logger import AudioLogger

# Import các Service PhoBERT
from whisper.phobert.phobert_service import PhobertService


class AudioPipeline:
    def __init__(self):
        print("[Pipeline] Đang khởi tạo các module AI (VAD, Whisper, PhoBERT)...")
        start_time = time.time()
        
        # Các module cũ
        self.vad = VADService()
        self.whisper = PhoWhisperService()
        self.detector = KeywordDetector()
        self.logger = AudioLogger()
        
        # Tích hợp thêm AI mới
        self.phobert = PhobertService()
        
        print(f"[Pipeline] Khởi tạo hoàn tất trong {time.time() - start_time:.2f}s")

    def process(self, audio_path: str):
        """Xử lý từ file wav/mp3"""
        audio = load_audio(audio_path)
        return self.process_audio(audio, source=audio_path)

    def process_audio(self, audio, timestamp=None, source="pipeline"):
        """Xử lý trực tiếp từ numpy array (microphone hoặc audio đã load)"""
        ts = float(timestamp if timestamp is not None else time.time())

        # ==========================================================
        # 1. Tiền xử lý & VAD (Gọt rác tần số + Loại bỏ khoảng lặng)
        # ==========================================================
        # Đảm bảo gọt sạch tiếng quạt laptop/tiếng phím Aula F75 trước khi vào VAD
        audio_filtered = apply_bandpass_filter(audio) 

        speech_segments = self.vad.detect_array(audio_filtered)
        speech_audio = extract_speech(audio_filtered, speech_segments)

        if len(speech_audio) == 0:
            return self._build_empty_response(ts, "idle")

        # ==========================================================
        # 2. Speech-to-Text (PhoWhisper)
        # ==========================================================
        transcript = self.whisper.transcribe(speech_audio)
        text = transcript.get("text", "").strip()

        # Bỏ qua nếu Whisper trả về chuỗi rỗng hoặc quá ngắn (ảo giác nhiễu)
        if not text or len(text) < 2:
            return self._build_empty_response(ts, "idle")

        # ==========================================================
        # 3. Phân tích Ngữ nghĩa (CẢI TIẾN: Lưới bảo vệ kép)
        # ==========================================================
        # 1. Lấy kết quả bắt từ khóa (Rule-based)
        keyword_result = self.detector.detect(text, timestamp=ts)
        rule_label = keyword_result.get("risk", "Normal").capitalize() 
        matched_keywords = keyword_result.get("matched", [])
        
        # 2. Lấy kết quả phân tích ngữ cảnh sâu (PhoBERT)
        ai_result = self.phobert.predict(text)
        phobert_label = ai_result.get("label", "Normal").capitalize()
        phobert_conf = ai_result.get("confidence", 0.0)
        
        # 3. TRỌNG TÀI QUYẾT ĐỊNH (Chỉ cần 1 trong 2 nghi ngờ là bắt)
        if rule_label in ["Suspicious", "Cheating"] or phobert_label in ["Suspicious", "Cheating"]:
            status_signal = "alert"
            final_risk = "Cheating" if "Cheating" in [rule_label, phobert_label] else "Suspicious"
            
            # Ưu tiên ghi nhận lý do từ khóa trước (vì nó chính xác với các câu ngắn)
            if rule_label in ["Suspicious", "Cheating"]:
                fusion_reason = f"Bắt được từ khóa cấm: {matched_keywords}"
            else:
                fusion_reason = f"AI PhoBERT phát hiện ngữ cảnh gian lận (Độ tự tin: {phobert_conf:.2f})"
        else:
            status_signal = "success"
            final_risk = "Normal"
            fusion_reason = f"Đánh giá an toàn (Độ tự tin: {phobert_conf:.2f})"

        # ==========================================================
        # 4. Logger (Ghi log thông minh hơn)
        # ==========================================================
        # Đã bỏ điều kiện IF, ghi lại toàn bộ lịch sử các câu thí sinh nói
        self.logger.write(
            text=text,
            risk=final_risk,
            confidence=phobert_conf,
            fusion_reason=fusion_reason,
            matched_keywords=matched_keywords,
            source=source,
            audio_length=len(speech_audio) / 16000.0,
        )

        # ==========================================================
        # 5. Final Result (Gọn gàng, sạch sẽ)
        # ==========================================================
        return {
            "module": "audio_pipeline",
            "status": status_signal, # Trả về chuẩn 'alert' / 'success' cho màn hình GUI
            "transcription": text,
            "speech_segments": speech_segments,
            "risk": final_risk,
            
            # Thông số AI chi tiết (Dùng cho Debug/Admin Dashboard)
            "confidence": round(phobert_conf, 4),
            "rule_label": rule_label,
            "ai_label": phobert_label,
            "fusion_reason": fusion_reason,
            
            # Giữ lại danh sách keyword bị bắt để Frontend bôi đỏ câu chữ
            "matched_keywords": matched_keywords, 
            "timestamp": ts,
        }

    def _build_empty_response(self, timestamp: float, status: str):
        """Hàm phụ trợ tạo JSON trả về khi không có tiếng người"""
        return {
            "module": "audio_pipeline",
            "status": status,
            "transcription": "",
            "speech_segments": [],
            "risk": "Normal",
            "confidence": 0.0,
            "rule_label": "Normal",
            "ai_label": "Normal",
            "fusion_reason": "No speech detected",
            "matched_keywords": [],
            "timestamp": timestamp,
        }