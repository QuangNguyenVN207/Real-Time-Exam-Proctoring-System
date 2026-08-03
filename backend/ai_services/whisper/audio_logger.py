from pathlib import Path
from datetime import datetime


class AudioLogger:

    def __init__(self):

        self.log_file = (
            Path(__file__).resolve().parent /
            "audio_log.txt"
        )

    def write(self, text, matched):

        now = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open(
            self.log_file,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(f"[{now}]\n")

            f.write(f"Text: {text}\n")

            f.write("Keyword:\n")

            for item in matched:

                f.write(
                    f" - {item['keyword']} "
                    f"(score={item['score']:.1f})\n"
                )

            f.write("\n")