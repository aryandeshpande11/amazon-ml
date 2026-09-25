# preprocessing.py
"""
Text normalization for business names and addresses.
Shared by blocking.py and features.py so candidate generation and
feature computation see identical strings.
"""
import re

LEGAL_SUFFIX_MAP = {
    "corporation": "corp", "corp.": "corp", "corp": "corp",
    "incorporated": "inc", "inc.": "inc", "inc": "inc",
    "limited": "ltd", "ltd.": "ltd", "ltd": "ltd",
    "private": "pvt", "pvt.": "pvt", "pvt": "pvt",
    "company": "co", "co.": "co", "co": "co",
    "llc": "llc", "llp": "llp", "plc": "plc",
}
STOPWORDS = {"the", "and", "of", "a", "an"}

ADDRESS_ABBR_MAP = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "apartment": "apt", "building": "bldg", "floor": "fl",
    "near": "near", "opposite": "opp", "behind": "behind",
}

PIN_RE = re.compile(r"\b\d{5,6}\b")
NUM_RE = re.compile(r"\d+")
NON_ALNUM_RE = re.compile(r"[^\w\s]")


def _tokenize(text: str) -> list:
    text = str(text).lower().replace("&", " and ")
    text = NON_ALNUM_RE.sub(" ", text)
    return text.split()


def normalize_name(name) -> str:
    """Lowercase, strip punctuation, expand/collapse legal suffixes, drop stopwords."""
    if name is None or (isinstance(name, float) and name != name):
        return ""
    tokens = _tokenize(name)
    tokens = [LEGAL_SUFFIX_MAP.get(t, t) for t in tokens]
    tokens = [t for t in tokens if t not in STOPWORDS and t]
    return " ".join(tokens)


def normalize_address(address) -> dict:
    """
    Returns a dict with:
      norm       - normalized address string (abbreviations expanded, landmark kept)
      pin        - postal/PIN code if found, else ""
      numbers    - set of numeric tokens found (house no., PIN, etc.)
      landmark   - substring following near/opposite/behind, if present
    """
    if address is None or (isinstance(address, float) and address != address):
        return {"norm": "", "pin": "", "numbers": set(), "landmark": ""}

    raw = str(address).lower()
    pin_match = PIN_RE.search(raw)
    pin = pin_match.group(0) if pin_match else ""
    numbers = set(NUM_RE.findall(raw))

    landmark = ""
    for kw in ("near", "opposite", "behind"):
        idx = raw.find(kw)
        if idx != -1:
            landmark = raw[idx:idx + 40].strip()
            break

    tokens = _tokenize(raw)
    tokens = [ADDRESS_ABBR_MAP.get(t, t) for t in tokens]
    norm = " ".join(tokens)
    return {"norm": norm, "pin": pin, "numbers": numbers, "landmark": landmark}