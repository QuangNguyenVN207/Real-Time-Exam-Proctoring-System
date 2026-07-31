import re
import unicodedata

# ==========================
# Chuẩn hóa số
# ==========================

NUMBER_MAP = {
    "mot": "1",
    "hai": "2",
    "ba": "3",
    "bon": "4",
    "tu": "4",
    "nam": "5",
    "sau": "6",
    "bay": "7",
    "tam": "8",
    "chin": "9",
    "muoi": "10",
}

# ==========================
# Chuẩn hóa đáp án A B C D
# ==========================

LETTER_MAP = {
    "a": "a",
    "be": "b",
    "b": "b",
    "xe": "c",
    "xe": "c",
    "c": "c",
    "de": "d",
    "d": "d",
}

# ==========================
# Từ đệm cần loại bỏ
# ==========================

STOPWORDS = {
    "a",
    "ah",
    "uh",
    "uhm",
    "um",
    "e",
    "o",
    "u",
    "am",
}

# ==========================
# Chuẩn hóa văn bản
# ==========================

def normalize_text(text: str) -> str:

    # chữ thường
    text = text.lower()

    # bỏ dấu tiếng Việt
    text = unicodedata.normalize("NFD", text)

    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    # bỏ dấu câu
    text = re.sub(r"[^\w\s]", " ", text)

    # bỏ khoảng trắng dư
    text = re.sub(r"\s+", " ", text)

    tokens = text.strip().split()

    normalized = []

    for token in tokens:

        # đổi số
        token = NUMBER_MAP.get(token, token)

        # đổi A B C D
        token = LETTER_MAP.get(token, token)

        # bỏ từ đệm
        if token in STOPWORDS:
            continue

        normalized.append(token)

    return " ".join(normalized)

# ==========================
# Sinh n-gram
# ==========================

def generate_ngrams(
    tokens,
    min_n=1,
    max_n=3
):

    result = {}

    for n in range(min_n, max_n + 1):

        result[n] = []

        for i in range(len(tokens) - n + 1):

            result[n].append(
                " ".join(tokens[i:i+n])
            )

    return result