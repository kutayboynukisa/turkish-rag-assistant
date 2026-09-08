"""Turkish-aware text normalization for search and retrieval."""

_TURKISH_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı"})


def normalize_turkish(text: str) -> str:
    """Lowercase text using Turkish casing rules.


    Python's default lowercasing follow English rules, which maps the
    dotless capital I to a dotted i. Turkish distinguishes the two: capital
    İ lowercases to i, and capital I lowercases to ı. This function applies
    the Turkish mapping before falling back to standard lowercasing, so
    other Turkish letters (ü, ö, ş, ğ, ç) are handled normally.


    Args:
        text: The input text to normalize.


    Returns:
        The lowercased text following Turkish casing rules.
    """
    return text.translate(_TURKISH_LOWER_MAP).lower()
