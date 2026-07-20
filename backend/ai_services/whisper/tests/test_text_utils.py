from whisper.text_utils import normalize_text

tests = [
    "Đáp Án!!!",
    "Đọc   đáp   án",
    "Cho xem câu số 5.",
    "ChatGPT giúp tao!!!",
    "Google???",
]

for t in tests:
    print(t)
    print(normalize_text(t))
    print("-" * 40)