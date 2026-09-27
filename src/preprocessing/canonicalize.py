"""Unicode-aware text canonicalization for entity resolution.

Preserves original fields and creates separate normalized fields.
Never removes non-ASCII characters from originals.

Decision D003: Raw and normalized fields are kept side-by-side.
"""

import re
import unicodedata

from unidecode import unidecode


# --------------------------------------------------------------------------- #
# Compiled regexes                                                             #
# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

# Legal suffixes to strip for blocking keys (not from original)
LEGAL_SUFFIXES = frozenset([
    "pvt", "private", "ltd", "limited", "llc", "inc", "incorporated",
    "corp", "corporation", "co", "company", "llp", "lp", "plc", "pllc",
    "sarl", "sas", "sa", "eurl", "sci", "snc", "gmbh", "ag",
    "limittedd", "praaivett", "praiveett",  # common misspellings in data
])

# Prefixes to strip (noise tokens that appear in S2/S3 names)
NOISE_PREFIXES = frozenset(["--", "+", "smt", "shri", "sri"])

# Postcode regex: India PIN (6 digits), US ZIP (5 digits), FR (5 digits)
_PC_RE = re.compile(r"(?<!\d)(\d{3}\s?\d{3}|\d{5}(?:-\d{4})?)(?!\d)")


def normalize_text(s: str) -> str:
    """Normalize text for comparison: lowercase, transliterate, strip punctuation.

    This is used for blocking keys and similarity features.
    The original text is always preserved separately.
    """
    if not s:
        return ""
    # Transliterate to ASCII for comparison
    s = unidecode(str(s)).lower()
    # Normalize ampersand
    s = s.replace("&", " and ")
    # Strip punctuation
    s = _PUNCT.sub(" ", s)
    # Collapse whitespace
    s = _WS.sub(" ", s).strip()
    return s


def normalize_name(s: str) -> str:
    """Normalize business name: lowercase, transliterate, strip legal suffixes."""
    norm = normalize_text(s)
    if not norm:
        return ""
    tokens = norm.split()
    # Strip leading noise
    while tokens and tokens[0] in NOISE_PREFIXES:
        tokens.pop(0)
    # Strip legal suffixes from end
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    # Strip surrounding brackets/parens from remaining tokens
    cleaned = []
    for t in tokens:
        t = t.strip("()[]{}\"'")
        if t:
            cleaned.append(t)
    return " ".join(cleaned) if cleaned else norm


def normalize_address(s: str) -> str:
    """Normalize business address for comparison."""
    return normalize_text(s)


def extract_postcode(addr: str) -> str:
    """Extract postcode-like pattern from address. Returns '' if none found."""
    m = _PC_RE.findall(str(addr))
    if not m:
        return ""
    # Take the last match (usually the postcode)
    return m[-1].replace(" ", "").split("-")[0]


def name_tokens(s: str) -> list:
    """Return normalized name tokens."""
    n = normalize_name(s)
    return n.split() if n else []


def first_name_token(s: str) -> str:
    """Return the first significant name token."""
    tokens = name_tokens(s)
    return tokens[0] if tokens else ""


def name_key(s: str) -> str:
    """Return a sorted-token key for blocking."""
    tokens = name_tokens(s)
    return " ".join(sorted(tokens)) if tokens else ""


def first_n_chars(s: str, n: int = 4) -> str:
    """Return first n characters of normalized name for blocking."""
    norm = normalize_name(s)
    return norm[:n] if norm else ""


def canonicalize_records(data):
    """Add normalized fields to dataframe. Preserves all original columns.

    Adds: name_norm, addr_norm, name_key, first_token, postcode
    """
    import pandas as pd

    df = data.copy()
    df["name_norm"] = df["business_name"].map(normalize_name)
    df["addr_norm"] = df["business_address"].map(normalize_address)
    df["first_token"] = df["name_norm"].map(lambda s: s.split()[0] if s else "")
    df["postcode"] = df["business_address"].map(extract_postcode)
    df["name_key"] = df["name_norm"].map(lambda s: " ".join(sorted(s.split())) if s else "")
    return df
