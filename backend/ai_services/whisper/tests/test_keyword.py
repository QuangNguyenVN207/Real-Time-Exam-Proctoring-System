from whisper.keyword_detector import KeywordDetector

detector = KeywordDetector()

texts = [

    "Cho tao đáp án câu số 5",

    "Lạp án là câu B",

    "Google giúp tao",

    "Mở ChatGPT",

    "Hôm nay trời đẹp",

    "Chào"

]

for text in texts:

    print("=" * 60)

    print(text)

    result = detector.detect(text)

    print(result)