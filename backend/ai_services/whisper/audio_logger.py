import json

from pathlib import Path
from datetime import datetime


class AudioLogger:

    def __init__(self):

        self.log_file = (
            Path(__file__).resolve().parent /
            "audio_log.jsonl"
        )

    def write(
        self,
        text,
        confidence,
        risk,
        matched,
        matched_rules,
        source="microphone"
    ):

        now = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        log = {

            "source": source,

            "time": now,

            "text": text,

            "confidence": confidence,

            "risk": risk,

            "matched_keywords": [

                {
                    "keyword": item["keyword"],
                    "candidate": item["candidate"],
                    "score": item["score"],
                    "severity": item["severity"],
                    "category": item["category"]
                }

                for item in matched

            ],

            "matched_rules": matched_rules

        }

        with open(
            self.log_file,
            "a",
            encoding="utf-8"
        ) as f:

            json.dump(
                log,
                f,
                ensure_ascii=False
            )

            f.write("\n")