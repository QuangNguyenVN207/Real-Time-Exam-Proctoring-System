import time
import numpy as np

# Các tiện ích tiền xử lý (Đã bao gồm hàm apply_bandpass_filter từ bước trước)
from backend.ai_services.whisper.audio_utils import load_audio, extract_speech, apply_bandpass_filter

# Import các AI Services hiện có
from backend.ai_services.whisper.vad_service import VADService
from backend.ai_services.whisper.phowhisper_service import PhoWhisperService
from backend.ai_services.whisper.keyword_detector import KeywordDetector
from backend.ai_services.whisper.audio_logger import AudioLogger

# Import các Service PhoBERT (giả định bạn đặt trong thư mục phobert)
from backend.ai_services.whisper.phobert.phobert_service import PhobertService
from backend.ai_services.whisper.phobert.decision_fusion import DecisionFusionService


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
        self.fusion = DecisionFusionService()
        
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
        # 3. Phân tích Ngữ nghĩa (Keyword + PhoBERT + Fusion)
        # ==========================================================
        # Lấy nhãn thô từ Keyword Detector
        keyword_result = self.detector.detect(text, timestamp=ts)
        rule_label = keyword_result.get("risk", "Normal").capitalize() # Đảm bảo format: Normal/Suspicious/Cheating
        
        # PhoBERT đánh giá ngữ cảnh sâu
        ai_result = self.phobert.predict(text)

        # Trọng tài Fusion quyết định cuối cùng
        fusion_result = self.fusion.fuse(text, rule_label, ai_result)
        final_risk = fusion_result["final_label"]

        # ==========================================================
        # 4. Logger (Ghi log thông minh hơn)
        # ==========================================================
        if final_risk in ["Suspicious", "Cheating"]:
            self.logger.write(
                text=text,
                risk=final_risk,
                confidence=ai_result["confidence"],
                fusion_reason=fusion_result["fusion_reason"],
                matched_keywords=keyword_result.get("matched", []),
                source=source,
                audio_length=len(speech_audio) / 16000.0,
            )

        # ==========================================================
        # 5. Final Result (Gọn gàng, sạch sẽ)
        # ==========================================================
        return {
            "module": "audio_pipeline",
            "status": final_risk,
            "transcription": text,
            "speech_segments": speech_segments,
            "risk": final_risk,
            
            # Thông số AI chi tiết (Dùng cho Debug/Admin Dashboard)
            "confidence": round(ai_result["confidence"], 4),
            "rule_label": rule_label,
            "ai_label": fusion_result["ai_label"],
            "fusion_reason": fusion_result["fusion_reason"],
            
            # Giữ lại danh sách keyword bị bắt để Frontend bôi đỏ câu chữ
            "matched_keywords": keyword_result.get("matched", []), 
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