import json
import re
import time
from collections import deque
from pathlib import Path

from rapidfuzz import fuzz

from whisper.text_utils import (
    normalize_text,
    generate_ngrams,
)
from whisper.rules import RULES
from whisper.negative_rules import NEGATIVE_RULES
from whisper.context_words import CONTEXT_WORDS


SAFE_SENTENCES = [
    "thay cho em hoi",
    "bat dau lam bai",
    "em xin phep",
    "em nop bai",
    "em lam xong roi",
    "cam on thay",
    "chao",
    "chao thay",
]


class KeywordDetector:
    # ==========================================================
    # Threshold tuning
    # ==========================================================
    KEYWORD_THRESHOLD_SINGLE = 92
    KEYWORD_THRESHOLD_MULTI = 90
    KEYWORD_THRESHOLD_AI_INTERNET = 95

    # ngưỡng alert sau khi cộng điểm
    ALERT_THRESHOLD = 85

    # ==========================================================
    # Time-window logic
    # ==========================================================
    WINDOW_SECONDS = 5.0
    WINDOW_MIN_HITS = 2

    def __init__(self):
        base_dir = Path(__file__).resolve().parent

        with open(base_dir / "keywords.json", encoding="utf-8") as f:
            raw_keywords = json.load(f)

        # deduplicate keywords
        seen = set()
        dedup = []
        for item in raw_keywords:
            key = str(item.get("keyword", "")).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            dedup.append(item)

        self.keywords = dedup

        self.alert_window = deque()
        self.last_text = ""

    # ==========================================================
    # Utils
    # ==========================================================
    def _normalize_severity(self, severity):
        if isinstance(severity, (int, float)):
            if severity >= 3:
                return "high"
            if severity >= 2:
                return "medium"
            return "low"

        return str(severity).strip().lower()

    def _regex_phrase_hit(self, text, phrase):
        phrase = str(phrase).strip()
        if not phrase:
            return False

        # text đã normalize => chỉ còn lowercase ASCII-ish + số + khoảng trắng
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
        return re.search(pattern, text) is not None

    def _threshold_for_item(self, keyword, item):
        token_count = len(keyword.split())
        category = str(item.get("category", "")).strip().lower()
        severity = self._normalize_severity(item.get("severity", "medium"))

        if category in {"ai", "internet"}:
            if token_count == 1:
                return self.KEYWORD_THRESHOLD_AI_INTERNET
            return 92

        if token_count == 1:
            return self.KEYWORD_THRESHOLD_SINGLE

        if severity == "high":
            return self.KEYWORD_THRESHOLD_MULTI
        if severity == "medium":
            return 88
        return 92

    def _cleanup_window(self, timestamp):
        while self.alert_window and (timestamp - self.alert_window[0]) > self.WINDOW_SECONDS:
            self.alert_window.popleft()

    def _register_alert_hit(self, timestamp):
        self.alert_window.append(timestamp)
        self._cleanup_window(timestamp)
        return len(self.alert_window) >= self.WINDOW_MIN_HITS

    # ==========================================================
    # Keyword Matching
    # ==========================================================
    def find_best_match(self, keyword, ngram_dict):
        token_count = len(keyword.split())
        candidates = ngram_dict.get(token_count, [])

        best_candidate = ""
        best_score = 0

        for gram in candidates:
            if not gram:
                continue

            # keyword 1 từ: KHÔNG dùng partial_ratio để giảm bắt nhầm
            if token_count == 1:
                score = max(
                    fuzz.ratio(keyword, gram),
                    fuzz.token_sort_ratio(keyword, gram),
                )
            else:
                score = max(
                    fuzz.ratio(keyword, gram),
                    fuzz.partial_ratio(keyword, gram),
                    fuzz.token_sort_ratio(keyword, gram),
                )

            # bỏ candidate quá ngắn
            if len(gram) < max(2, int(len(keyword) * 0.6)):
                continue

            if score > best_score:
                best_score = score
                best_candidate = gram

        return best_candidate, best_score

    # ==========================================================
    # Rule Score
    # ==========================================================
    def rule_score(self, text):
        score = 0
        matched_rules = []

        for rule in RULES:
            rule_name = rule.get("name", "unknown")
            rule_score = int(rule.get("score", 0))

            if "contains_all" in rule:
                ok = True
                for phrase in rule["contains_all"]:
                    if not self._regex_phrase_hit(text, phrase):
                        ok = False
                        break

                if ok:
                    score += rule_score
                    matched_rules.append(rule_name)

            elif "contains_any" in rule:
                ok = False
                for phrase in rule["contains_any"]:
                    if self._regex_phrase_hit(text, phrase):
                        ok = True
                        break

                if ok:
                    score += rule_score
                    matched_rules.append(rule_name)

            elif "contains" in rule:
                ok = True
                for phrase in rule["contains"]:
                    if not self._regex_phrase_hit(text, phrase):
                        ok = False
                        break

                if ok:
                    score += rule_score
                    matched_rules.append(rule_name)

        return score, matched_rules

    # ==========================================================
    # Context Score
    # ==========================================================
    def context_score(self, text):
        score = 0
        matched_context = []

        for word, value in CONTEXT_WORDS.items():
            if self._regex_phrase_hit(text, word):
                score += int(value)
                matched_context.append(word)

        return score, matched_context

    # ==========================================================
    # Negative Score
    # ==========================================================
    def negative_score(self, text):
        penalty = 0
        matched_negative = []

        for rule in NEGATIVE_RULES:
            rule_name = rule.get("name", "unknown")
            penalty_value = int(rule.get("penalty", 0))

            phrases = rule.get("contains_any") or rule.get("contains") or []

            hit = False
            for phrase in phrases:
                if self._regex_phrase_hit(text, phrase):
                    hit = True
                    break

            if hit:
                penalty += penalty_value
                matched_negative.append(rule_name)

        return penalty, matched_negative

    # ==========================================================
    # Risk
    # ==========================================================
    def get_risk(self, confidence):
        if confidence >= 90:
            return "high"
        if confidence >= 60:
            return "medium"
        if confidence >= 35:
            return "low"
        return "safe"

    # ==========================================================
    # Detect
    # ==========================================================
    def detect(self, text: str, timestamp=None):
        normalized_text = normalize_text(text)

        if timestamp is None:
            timestamp = time.time()
        else:
            timestamp = float(timestamp)

        if not normalized_text:
            return {
                "alert": False,
                "raw_alert": False,
                "confidence": 0,
                "score": 0,
                "risk": "safe",
                "keyword_score": 0,
                "rule_bonus": 0,
                "context_bonus": 0,
                "penalty": 0,
                "matched": [],
                "matched_rules": [],
                "matched_context": [],
                "matched_negative": [],
                "window_hits": len(self.alert_window),
                "window_seconds": self.WINDOW_SECONDS,
            }

        # chống lặp cùng một text liên tục
        if normalized_text == self.last_text:
            self._cleanup_window(timestamp)
            return {
                "alert": False,
                "raw_alert": False,
                "confidence": 0,
                "score": 0,
                "risk": "safe",
                "keyword_score": 0,
                "rule_bonus": 0,
                "context_bonus": 0,
                "penalty": 0,
                "matched": [],
                "matched_rules": [],
                "matched_context": [],
                "matched_negative": [],
                "window_hits": len(self.alert_window),
                "window_seconds": self.WINDOW_SECONDS,
            }

        self.last_text = normalized_text

        # whitelist
        for safe in SAFE_SENTENCES:
            if fuzz.ratio(normalized_text, safe) >= 90:
                self.alert_window.clear()
                return {
                    "alert": False,
                    "raw_alert": False,
                    "confidence": 0,
                    "score": 0,
                    "risk": "safe",
                    "keyword_score": 0,
                    "rule_bonus": 0,
                    "context_bonus": 0,
                    "penalty": 0,
                    "matched": [],
                    "matched_rules": [],
                    "matched_context": [],
                    "matched_negative": [],
                    "window_hits": len(self.alert_window),
                    "window_seconds": self.WINDOW_SECONDS,
                }

        tokens = normalized_text.split()
        ngram_dict = generate_ngrams(tokens, min_n=1, max_n=4)

        matched = []
        matched_keywords = set()
        keyword_score = 0

        for item in self.keywords:
            keyword = normalize_text(str(item.get("keyword", "")))
            if not keyword:
                continue

            # exact regex hit trước để giảm bắt nhầm
            if self._regex_phrase_hit(normalized_text, keyword):
                candidate, score = keyword, 100
            else:
                candidate, score = self.find_best_match(keyword, ngram_dict)

            threshold = self._threshold_for_item(keyword, item)

            if score < threshold:
                continue

            dedup_key = keyword
            if dedup_key in matched_keywords:
                continue
            matched_keywords.add(dedup_key)

            matched.append({
                "keyword": item.get("keyword", ""),
                "candidate": candidate,
                "score": round(score, 1),
                "severity": item.get("severity", ""),
                "category": item.get("category", ""),
            })

            keyword_score = max(keyword_score, score)

        matched.sort(key=lambda x: x["score"], reverse=True)

        rule_bonus, matched_rules = self.rule_score(normalized_text)
        context_bonus, matched_context = self.context_score(normalized_text)
        penalty, matched_negative = self.negative_score(normalized_text)

        raw_confidence = keyword_score + rule_bonus + context_bonus - penalty
        confidence = max(0, min(100, int(raw_confidence)))
        risk = self.get_risk(confidence)

        raw_alert = confidence >= self.ALERT_THRESHOLD

        # time-window logic: chỉ alert nếu có >= 2 lần trong 5 giây
        if raw_alert:
            alert = self._register_alert_hit(timestamp)
        else:
            self._cleanup_window(timestamp)
            alert = False

        return {
            "alert": alert,                 # confirmed alert (sau time-window)
            "raw_alert": raw_alert,         # alert thô trước time-window
            "confidence": confidence,
            "score": confidence,            # alias để tương thích code cũ
            "risk": risk,
            "keyword_score": round(keyword_score, 1),
            "rule_bonus": rule_bonus,
            "context_bonus": context_bonus,
            "penalty": penalty,
            "matched": matched[:5],
            "matched_rules": matched_rules,
            "matched_context": matched_context,
            "matched_negative": matched_negative,
            "window_hits": len(self.alert_window),
            "window_seconds": self.WINDOW_SECONDS,
        }