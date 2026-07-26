import json
from pathlib import Path

from rapidfuzz import fuzz

from whisper.text_utils import (
    normalize_text,
    generate_ngrams
)

from whisper.rules import RULES
from whisper.negative_rules import NEGATIVE_RULES
from whisper.context_words import CONTEXT_WORDS


class KeywordDetector:

    KEYWORD_THRESHOLD = 80

    def __init__(self):

        base_dir = Path(__file__).resolve().parent

        with open(
            base_dir / "keywords.json",
            encoding="utf-8"
        ) as f:

            self.keywords = json.load(f)

    # =======================================================
    # Keyword Matching
    # =======================================================

    def find_best_match(self, keyword, ngram_dict):

        token_count = len(keyword.split())

        candidates = ngram_dict.get(token_count, [])

        best_candidate = ""
        best_score = 0

        for gram in candidates:

            score = max(

                fuzz.ratio(keyword, gram),

                fuzz.partial_ratio(keyword, gram),

                fuzz.token_sort_ratio(keyword, gram)

            )

            if score > best_score:

                best_score = score
                best_candidate = gram

        return best_candidate, best_score

    # =======================================================
    # Rule Score
    # =======================================================

    def rule_score(self, text):

        score = 0

        matched_rules = []

        for rule in RULES:

            # contains_all

            if "contains_all" in rule:

                ok = True

                for word in rule["contains_all"]:

                    if word not in text:

                        ok = False
                        break

                if ok:

                    score += rule["score"]

                    matched_rules.append(rule["name"])

            # contains_any

            elif "contains_any" in rule:

                for word in rule["contains_any"]:

                    if word in text:

                        score += rule["score"]

                        matched_rules.append(rule["name"])

                        break

        return score, matched_rules

    # =======================================================
    # Context Score
    # =======================================================

    def context_score(self, text):

        score = 0

        matched_context = []

        for word, value in CONTEXT_WORDS.items():

            if word in text:

                score += value

                matched_context.append(word)

        return score, matched_context

    # =======================================================
    # Negative Rules
    # =======================================================

    def negative_score(self, text):

        penalty = 0

        matched_negative = []

        for rule in NEGATIVE_RULES:

            for sentence in rule["contains_any"]:

                if sentence in text:

                    penalty += rule["penalty"]

                    matched_negative.append(rule["name"])

                    break

        return penalty, matched_negative

    # =======================================================
    # Risk
    # =======================================================

    def get_risk(self, confidence):

        if confidence >= 90:

            return "high"

        if confidence >= 60:

            return "medium"

        if confidence >= 35:

            return "low"

        return "safe"

    # =======================================================
    # Detect
    # =======================================================

    def detect(self, text: str):

        normalized_text = normalize_text(text)

        tokens = normalized_text.split()

        ngram_dict = generate_ngrams(tokens)

        matched_keywords = []

        keyword_score = 0

        # ------------------------------------------

        for item in self.keywords:

            keyword = normalize_text(item["keyword"])

            candidate, score = self.find_best_match(
                keyword,
                ngram_dict
            )

            if score >= self.KEYWORD_THRESHOLD:

                matched_keywords.append({

                    "keyword": item["keyword"],

                    "candidate": candidate,

                    "score": round(score, 1),

                    "severity": item["severity"],

                    "category": item["category"]

                })

                keyword_score = max(
                    keyword_score,
                    score
                )

        # ------------------------------------------

        rule_bonus, matched_rules = self.rule_score(
            normalized_text
        )

        context_bonus, matched_context = self.context_score(
            normalized_text
        )

        penalty, matched_negative = self.negative_score(
            normalized_text
        )

        # ------------------------------------------

        confidence = (
            keyword_score
            + rule_bonus
            + context_bonus
            - penalty
        )

        confidence = max(
            0,
            min(100, int(confidence))
        )

        risk = self.get_risk(
            confidence
        )

        return {

            "alert": confidence >= 60,

            "confidence": confidence,

            "risk": risk,

            "keyword_score": round(keyword_score, 1),

            "rule_bonus": rule_bonus,

            "context_bonus": context_bonus,

            "penalty": penalty,

            "matched": matched_keywords,

            "matched_rules": matched_rules,

            "matched_context": matched_context,

            "matched_negative": matched_negative

        }