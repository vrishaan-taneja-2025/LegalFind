
import re

from snowballstemmer import stemmer

_ENGLISH_STEMMER = stemmer("english")


def tokenize(text: str) -> list[str]:
    """Lowercase, tokenize, and English Snowball-stem text consistently."""
    return _ENGLISH_STEMMER.stemWords(
        re.findall(r"[a-z0-9]+", text.casefold())
    )
