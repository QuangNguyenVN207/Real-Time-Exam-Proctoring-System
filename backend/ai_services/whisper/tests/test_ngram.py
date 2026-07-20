from whisper.text_utils import normalize_text, generate_ngrams

text = "Cho tao đáp án câu số 5"

normalized = normalize_text(text)

tokens = normalized.split()

ngrams = generate_ngrams(tokens)

print("=" * 50)
print("Normalized:")
print(normalized)

print("\nTokens:")
print(tokens)

print(f"\nCó {len(ngrams)} n-grams:\n")

for i, gram in enumerate(ngrams, start=1):
    print(f"{i:02d}. {gram}")