import os
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

#: Prompt miền thi cử cho PhoWhisper. Đo trên data/test_audio (6 mẫu): không cải thiện rõ rệt
#: nên MẶC ĐỊNH TẮT — prompt sai làm model thiên vị sang từ vựng trong prompt. Bật bằng
#: `PROCTOR_ASR_PROMPT="..."` nếu âm thanh của bạn toàn thuật ngữ thi cử.
ASR_PROMPT = os.environ.get("PROCTOR_ASR_PROMPT", "").strip()
#: Đoạn VAD ngắn hơn mức này thì không đưa vào Whisper. Mẩu 0.7 s tiếng ho/va chạm vẫn qua
#: VAD và Whisper sẽ "viết báo" thành cả câu (đã gặp: 0.75 s nhiễu -> 'là tay bà mẹ đứng về.').
ASR_MIN_SPEECH_S = float(os.environ.get("PROCTOR_ASR_MIN_SPEECH_S", "0.8"))
#: Cổng thứ hai: độ tự tin tiếng người trung bình của chính đoạn đó. Nói thật 0.76-0.85,
#: nhiễu lọt lưới 0.08-0.13 (đo 2026-10-29) -> 0.5 nằm giữa hai vùng, còn dư rất rộng.
ASR_MIN_SPEECH_PROB = float(os.environ.get("PROCTOR_ASR_MIN_SPEECH_PROB", "0.5"))
#: Mức RMS đưa vào Whisper. File quay bằng điện thoại ở xa có giọng nói RMS ~0.01 trong khi
#: ngưỡng cắt câu 0.02: khuếch đại cục bộ giúp model nghe rõ hơn mà không đổi ngưỡng chung.
ASR_TARGET_RMS = 0.1
ASR_PEAK_CEILING = 0.95
ASR_SILENT_RMS = 0.003


def _normalize_level(audio: np.ndarray) -> np.ndarray:
    """Khuếch đại đoạn tiếng về mức ASR_TARGET_RMS, chặn đỉnh; không bao giờ giảm âm lượng."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < ASR_SILENT_RMS:
        return audio
    gain = ASR_TARGET_RMS / rms
    peak = float(np.abs(audio).max())
    if peak > 0:
        gain = min(gain, ASR_PEAK_CEILING / peak)
    if gain <= 1.0:
        return audio
    return (audio * gain).astype(np.float32)


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

    def process_audio(self, audio, timestamp=None, source="pipeline", bypass_gate=False):
        """Xử lý trực tiếp từ numpy array (microphone hoặc audio đã load).

        `bypass_gate=True` tắt cổng chặn ảo giác — chỉ dùng cho ablation trong
        `scripts/asr_ablation.py` để đo tỉ lệ câu bịa khi không có cổng.
        """
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

        speech_seconds = len(speech_audio) / 16000.0
        speech_prob = self.vad.mean_probability(speech_audio)
        reason = None
        if bypass_gate:
            reason = None
        elif speech_seconds < ASR_MIN_SPEECH_S:
            reason = f"dài {speech_seconds:.2f}s < {ASR_MIN_SPEECH_S}s"
        elif speech_prob < ASR_MIN_SPEECH_PROB:
            reason = (f"độ tự tin tiếng người {speech_prob:.2f} < {ASR_MIN_SPEECH_PROB} "
                      f"(nói rõ dù lẫn nhiễu đo được 0.55-0.85)")
        if reason:
            print(f"[AUDIO] Bỏ đoạn {speech_seconds:.2f}s — {reason}: đưa vào Whisper chỉ tổ bịa câu.")
            return self._build_empty_response(ts, "idle")

        speech_audio = _normalize_level(speech_audio)

        # ==========================================================
        # 2. Speech-to-Text (PhoWhisper)
        # ==========================================================
        transcript = self.whisper.transcribe(speech_audio, prompt=ASR_PROMPT or None)
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