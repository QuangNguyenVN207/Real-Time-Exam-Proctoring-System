import re
import unicodedata


def normalize_text(text: str):

    text = text.lower()

    text = unicodedata.normalize("NFD", text)

    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def generate_ngrams(tokens, max_n=3):

    result = {}

    for n in range(1, max_n + 1):

        result[n] = []

        for i in range(len(tokens) - n + 1):

            result[n].append(
                " ".join(tokens[i:i+n])
            )

    return result