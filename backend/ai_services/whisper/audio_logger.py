import json
from pathlib import Path
from datetime import datetime


class AudioLogger:
    def __init__(self):
        self.log_file = Path(__file__).resolve().parent / "audio_log.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        text,
        confidence,
        risk,
        matched=None,
        matched_rules=None,
        source="microphone",
        processing_time=None,
        audio_length=None,
    ):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        matched = matched or []
        matched_rules = matched_rules or []

        log = {
            "source": source,
            "time": now,
            "text": text,
            "confidence": confidence,
            "risk": risk,
            "matched_keywords": [
                {
                    "keyword": item.get("keyword", ""),
                    "candidate": item.get("candidate", ""),
                    "score": item.get("score", 0),
                    "severity": item.get("severity", ""),
                    "category": item.get("category", ""),
                }
                for item in matched
            ],
            "matched_rules": matched_rules,
        }

        if processing_time is not None:
            log["processing_time"] = processing_time

        if audio_length is not None:
            log["audio_length"] = audio_length

        with open(self.log_file, "a", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False)
            f.write("\n")