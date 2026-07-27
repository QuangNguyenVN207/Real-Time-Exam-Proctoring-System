import json

from pathlib import Path

from rapidfuzz import fuzz

from whisper.text_utils import (
    normalize_text,
    generate_ngrams
)

SAFE_SENTENCES = [
    "thay cho em hoi",
    "bat dau lam bai",
    "em xin phep",
    "em nop bai",
    "em lam xong roi",
    "cam on thay",
    "chao",
    "chao thay"
]


class KeywordDetector:

    def __init__(self):

        base_dir = Path(__file__).resolve().parent

        with open(
            base_dir / "keywords.json",
            encoding="utf-8"
        ) as f:

            self.keywords = json.load(f)

    def token_similarity(self, keyword, candidate):

        keyword_tokens = keyword.split()
        candidate_tokens = candidate.split()

        if len(keyword_tokens) != len(candidate_tokens):
            return 0

        scores = []

        for kw, cd in zip(keyword_tokens, candidate_tokens):

            scores.append(
                fuzz.ratio(kw, cd)
            )

        return sum(scores) / len(scores)

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
        # Kiểm tra các câu hợp lệ (whitelist)
        if normalized_text in SAFE_SENTENCES:
            return {
                "alert": False,
                "score": 0,
                "matched": []
            }
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

                print(
                    f"[MATCH] '{candidate}' -> '{item['keyword']}' ({score:.1f})"
                )

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