from whisper.keyword_detector import KeywordDetector
from whisper.text_utils import normalize_text, generate_ngrams

text = "Cho tao đáp án câu số 5"

detector = KeywordDetector()

normalized = normalize_text(text)
tokens = normalized.split()
ngrams = generate_ngrams(tokens)

candidate, score = detector.find_best_match(
    "dap an",
    ngrams
)

print("Best candidate:", candidate)
print("Score:", score)