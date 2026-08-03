import json
from pathlib import Path
from datetime import datetime

class AudioLogger:
    def __init__(self, log_dir="outputs"):
        """
        Sử dụng pathlib (từ file cũ) để quản lý đường dẫn đa nền tảng tốt hơn.
        Mặc định lưu vào thư mục 'outputs' để tách biệt log với code source.
        """
        # Trỏ ra ngoài thư mục gọi script hoặc sử dụng thư mục outputs
        base_dir = Path(__file__).resolve().parent.parent
        self.log_file = base_dir / log_dir / "audio_log.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        text,
        confidence,
        risk,
        fusion_reason="",        # Tham số mới từ Decision Fusion
        matched_keywords=None,   # Tham số mới thay thế cho matched
        matched=None,            # Giữ lại để tương thích ngược với code cũ
        matched_rules=None,      # Giữ lại để tương thích ngược
        source="microphone",
        processing_time=None,
        audio_length=None,
        **kwargs                 # 🛡️ Hứng mọi tham số dư thừa để chống crash hệ thống
    ):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ưu tiên dữ liệu từ keyword mới, nếu không có thì fallback về code cũ
        actual_keywords = matched_keywords if matched_keywords is not None else (matched or [])
        actual_rules = matched_rules or []

        # Đóng gói dữ liệu log
        log = {
            "time": now,
            "source": source,
            "text": text,
            "risk": risk,
            "confidence": round(confidence, 4) if isinstance(confidence, (int, float)) else confidence,
            "fusion_reason": fusion_reason,
            "matched_keywords": [
                {
                    "keyword": item.get("keyword", ""),
                    "candidate": item.get("candidate", ""),
                    "score": item.get("score", 0),
                    "severity": item.get("severity", ""),
                    "category": item.get("category", ""),
                }
                for item in actual_keywords
            ],
            "matched_rules": actual_rules,
        }

        # Nếu có đo lường thời gian thì đưa vào log
        if processing_time is not None:
            log["processing_time_sec"] = round(processing_time, 3)

        if audio_length is not None:
            log["audio_length_sec"] = round(audio_length, 2)

        # Ghi log dạng JSON Lines (mỗi json 1 dòng, an toàn không lo hỏng file nếu cúp điện)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            print(f"[Logger Error] Lỗi khi ghi log: {str(e)}")