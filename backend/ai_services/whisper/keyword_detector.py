import json

from pathlib import Path

from rapidfuzz import fuzz

from whisper.text_utils import (
    normalize_text,
    generate_ngrams
)


class KeywordDetector:

    def __init__(self):

        base_dir = Path(__file__).resolve().parent

        with open(
            base_dir / "keywords.json",
            encoding="utf-8"
        ) as f:

            self.keywords = json.load(f)

    def find_best_match(self, keyword, ngram_dict):

        token_count = len(keyword.split())

        candidates = ngram_dict.get(token_count, [])

        best_candidate = ""
        best_score = 0

        for gram in candidates:

            score = max(
                fuzz.ratio(keyword, gram),
                fuzz.token_sort_ratio(keyword, gram)
            )

            if score > best_score:

                best_score = score
                best_candidate = gram

        return best_candidate, best_score

    def detect(self, text: str):

        normalized_text = normalize_text(text)

        tokens = normalized_text.split()

        ngram_dict = generate_ngrams(tokens)

        matched = []

        best_score = 0

        for item in self.keywords:

            keyword = normalize_text(
                item["keyword"]
            )

            candidate, score = self.find_best_match(
                keyword,
                ngram_dict
            )

            if score >= 80:

                matched.append({

                    "keyword": item["keyword"],

                    "candidate": candidate,

                    "score": score,

                    "severity": item["severity"],

                    "category": item["category"]

                })

                best_score = max(
                    best_score,
                    score
                )

        return {

            "alert": len(matched) > 0,

            "score": best_score,

            "matched": matched

        }