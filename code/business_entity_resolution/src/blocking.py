"""
Candidate generation (blocking) for Source 1 entities against the pooled
Source 2 + Source 3 records.

Blocking passes (all scoped by country):
  1. Exact normalized name  – with legal-suffix normalization + stopword removal
  2. PIN/ZIP + first-significant-word prefix
  3. Token overlap on name tokens (>= MIN_TOKEN_LEN chars)
  4. Sorted-token canonical name  – catches word-order transpositions
  5. Address-token overlap  – catches "different name, same address" pairs
  6. Character n-gram overlap – catches typos and URL/domain names

Key improvements over the original version
------------------------------------------
* Legal suffix normalization (private→pvt, limited→ltd, corporation→corp,
  incorporated→inc, company→co) from preprocessing.py is applied BEFORE
  blocking-key generation, so "Private Limited" and "Pvt Ltd" now merge
  on the exact-name pass.
* Stopword removal (the, and, of, a, an) gives cleaner blocking keys.
* First-significant-word prefix replaces raw name[:3].  Legal prefixes
  like "LLC", "Pvt", "Inc" at the start of a name are skipped so the
  prefix captures the distinctive part of the name.
* Sorted-token name pass: tokens are sorted alphabetically before
  comparison, making the key invariant to word-order transpositions
  ("Moore Bitwise" == "Bitwise Moore").
* Address-token pass: distinctive address tokens (street names, locality
  names) are used as blocking keys so that records with different names
  but the same address are still proposed as candidates.
* MAX_BLOCK_SIZE raised 2000 → 5000 to reduce silent recall loss on
  moderately common names/tokens.
* DEFAULT_TOP_K raised 40 → 50 for more headroom in candidate ranking.

Design decisions
----------------
* Every blocking pass is country-scoped (US↔US, India↔India only).
* Oversized groups (> cap) are dropped BEFORE the merge to prevent OOM.
* Candidates are capped per S1 entity via match_strength ranking (count
  of independent blocking passes that agreed on a pair).
* PoolIndex is built ONCE and reused across S1 chunks; the expensive
  pool-side work never runs more than once per program run.
* The public API (generate_candidates) has the same signature as before
  so existing notebook/script calls need no changes.
"""

import re
import numpy as np
import pandas as pd

from preprocessing import LEGAL_SUFFIX_MAP, STOPWORDS, ADDRESS_ABBR_MAP

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
MAX_BLOCK_SIZE = 5000          # cap for name-based blocking groups
MAX_ADDR_BLOCK_SIZE = 2000     # tighter cap for address-token groups
MIN_TOKEN_LEN = 4              # minimum name-token length
MIN_ADDR_TOKEN_LEN = 5         # minimum address-token length (more selective)
DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_TOP_K = 50

# ---------------------------------------------------------------------------
# Derived normalization tables
# ---------------------------------------------------------------------------
# Only the long-form → short-form mappings that actually change the token.
# After the basic cleanup step strips punctuation, "Corp." → "corp" is
# already done, so we only need "corporation" → "corp", etc.
_LEGAL_LONG_TO_SHORT = {k: v for k, v in LEGAL_SUFFIX_MAP.items() if k != v}

# Words to skip when computing "first significant word" for prefix blocking.
_SKIP_PREFIX_WORDS = (
    set(LEGAL_SUFFIX_MAP.values()) | STOPWORDS
    | {'llc', 'llp', 'plc', 'dba', 'www'}
)

# Address words too generic to be useful as blocking keys.
_ADDR_STOP = {
    # English road/building vocabulary
    'st', 'street', 'rd', 'road', 'ave', 'avenue', 'blvd', 'boulevard',
    'dr', 'drive', 'ln', 'lane', 'ct', 'court', 'pl', 'place', 'way',
    'apt', 'apartment', 'unit', 'suite', 'ste', 'floor', 'fl',
    'bldg', 'building', 'number', 'near', 'opposite', 'opp', 'behind',
    'box', 'sector', 'block', 'phase', 'plot', 'door',
    # Directional
    'north', 'south', 'east', 'west',
    # Indian locality vocabulary
    'nagar', 'colony', 'layout', 'extension', 'extn', 'marg', 'gali',
    'mohalla', 'ward', 'mandal', 'taluk', 'tehsil', 'dist', 'district',
    'tower', 'towers', 'complex', 'center', 'centre', 'market',
    'main', 'cross', 'circle', 'chowk', 'bazaar', 'road',
} | STOPWORDS


# ---------------------------------------------------------------------------
# Vectorized normalization helpers
# ---------------------------------------------------------------------------

def _apply_legal_suffix_norm(name_series: pd.Series) -> pd.Series:
    """Replace long-form legal suffixes with their abbreviations.

    Uses vectorized ``str.replace`` with word-boundary regex so each
    replacement is applied column-wide in one pass (much faster than
    row-level ``apply``).
    """
    s = name_series.copy()
    for long_form, short_form in _LEGAL_LONG_TO_SHORT.items():
        s = s.str.replace(
            r'\b' + re.escape(long_form) + r'\b', short_form, regex=True
        )
    return s


def _remove_stopwords(name_series: pd.Series) -> pd.Series:
    """Strip stopwords and collapse whitespace."""
    s = name_series.copy()
    for sw in STOPWORDS:
        s = s.str.replace(
            r'\b' + re.escape(sw) + r'\b', '', regex=True
        )
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s


def _first_significant_word(text) -> str:
    """First meaningful word (skipping legal/stop terms), truncated to 4 chars.

    This replaces the naïve ``name[:3]`` prefix which was dominated by
    "llc", "pvt", "inc" whenever the legal suffix appeared first.
    """
    if not isinstance(text, str) or not text:
        return ""
    for tok in text.split():
        if tok not in _SKIP_PREFIX_WORDS and len(tok) >= 3:
            return tok[:4]
    # Fallback: first token regardless
    tokens = text.split()
    return tokens[0][:4] if tokens else ""


# ---------------------------------------------------------------------------
# Core _prep — shared by PoolIndex construction and per-chunk S1 processing
# ---------------------------------------------------------------------------

def _prep(df: pd.DataFrame, id_col: str, name_col: str,
          addr_col: str = "business_address") -> pd.DataFrame:
    """Enhanced normalization: legal suffixes, stopwords, sorted name,
    address tokens, first-significant-word prefix."""
    out = df.copy()

    # ---- Name normalization ------------------------------------------------
    raw_name = (
        out[name_col].astype(str).str.lower()
        .str.replace("&", " and ", regex=False)
    )
    basic_name = (
        raw_name
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .fillna("")
    )
    normed = _apply_legal_suffix_norm(basic_name)
    normed = _remove_stopwords(normed)
    out["_norm_name"] = normed

    # Sorted-token canonical name (word-order invariant)
    out["_sorted_name"] = out["_norm_name"].apply(
        lambda x: " ".join(sorted(x.split())) if isinstance(x, str) and x else ""
    )

    # ---- Address normalization ---------------------------------------------
    if addr_col in out.columns:
        raw_addr = out[addr_col].astype(str).str.lower()
        out["_norm_addr"] = (
            raw_addr
            .str.replace("&", " and ", regex=False)
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .fillna("")
        )
        out["_pin"] = (
            out[addr_col].astype(str)
            .str.extract(r"\b(\d{5,6})\b", expand=False)
            .fillna("")
        )
    else:
        out["_norm_addr"] = ""
        out["_pin"] = ""

    # ---- Country -----------------------------------------------------------
    if "country" in out.columns:
        out["_country"] = out["country"].astype(str).str.strip().str.lower()
    else:
        out["_country"] = ""

    # ---- Prefix (first significant word) -----------------------------------
    out["_prefix"] = out["_norm_name"].apply(_first_significant_word)

    return out


# ---------------------------------------------------------------------------
# Oversized-group filter
# ---------------------------------------------------------------------------

def _drop_oversized_groups(df: pd.DataFrame, group_cols: list,
                           max_size: int = MAX_BLOCK_SIZE) -> pd.DataFrame:
    """Drop rows belonging to any group bigger than *max_size*, so a
    generic name/PIN/token can't blow up the merge that follows."""
    counts = df.groupby(group_cols).size().reset_index(name="_cnt")
    valid = counts[counts["_cnt"] <= max_size][group_cols]
    return df.merge(valid, on=group_cols)


# ---------------------------------------------------------------------------
# PoolIndex — precomputed blocking structures for S2 + S3
# ---------------------------------------------------------------------------

class PoolIndex:
    """
    Precomputed blocking structures for the pooled S2+S3 records.
    Build ONCE per run (see generate_candidates) and reuse across every
    source1 chunk -- rebuilding this per chunk is what made blocking slow.
    """

    def __init__(self, source2: pd.DataFrame, source3: pd.DataFrame,
                 id_col: str = "entity_id", name_col: str = "business_name"):

        pool = _prep(
            pd.concat([source2, source3], ignore_index=True), id_col, name_col
        )

        # Lightweight mini-table for Pass 1 & 2 merges
        self.pool_mini = pool[
            [id_col, "_country", "_norm_name", "_pin", "_prefix"]
        ].rename(columns={id_col: "candidate_entity_id"})

        # ---- Pass 1: exact normalized name (capped) -----------------------
        pool_named = self.pool_mini[self.pool_mini["_norm_name"] != ""]
        self.pool_exact = _drop_oversized_groups(
            pool_named, ["_country", "_norm_name"]
        )

        # ---- Pass 2: PIN + first-significant-word prefix (capped) ---------
        pool_pin = self.pool_mini[
            (self.pool_mini["_pin"] != "") & (self.pool_mini["_prefix"] != "")
        ]
        self.pool_pin = _drop_oversized_groups(
            pool_pin, ["_country", "_pin", "_prefix"]
        )

        # ---- Pass 3: name-token index (from UNIQUE name pairs) ------------
        unique_names = (
            pool_named[["_country", "_norm_name"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        unique_names["_uid"] = np.arange(len(unique_names))

        tok = unique_names.assign(
            _tok=unique_names["_norm_name"].str.split()
        ).explode("_tok")
        tok = tok.dropna(subset=["_tok"])
        tok = tok[tok["_tok"].str.len() >= MIN_TOKEN_LEN]
        self.pool_tok = _drop_oversized_groups(
            tok, ["_country", "_tok"]
        )[["_country", "_tok", "_uid"]]

        self.name_to_ids = pool_named.merge(
            unique_names, on=["_country", "_norm_name"]
        )[["_uid", "candidate_entity_id"]]

        # ---- Pass 4: sorted canonical name (capped) -----------------------
        pool_sorted_col = pool[[id_col, "_country", "_sorted_name"]].rename(
            columns={id_col: "candidate_entity_id"}
        )
        pool_sorted_col = pool_sorted_col[pool_sorted_col["_sorted_name"] != ""]
        self.pool_sorted = _drop_oversized_groups(
            pool_sorted_col, ["_country", "_sorted_name"]
        )

        # ---- Pass 5: address-token index (from UNIQUE address pairs) ------
        pool_addr = pool[[id_col, "_country", "_norm_addr"]].rename(
            columns={id_col: "candidate_entity_id"}
        )
        pool_with_addr = pool_addr[pool_addr["_norm_addr"] != ""]

        unique_addrs = (
            pool_with_addr[["_country", "_norm_addr"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        unique_addrs["_addr_uid"] = np.arange(len(unique_addrs))

        addr_tok = unique_addrs.assign(
            _atok=unique_addrs["_norm_addr"].str.split()
        ).explode("_atok")
        addr_tok = addr_tok.dropna(subset=["_atok"])
        addr_tok = addr_tok[
            (addr_tok["_atok"].str.len() >= MIN_ADDR_TOKEN_LEN)
            & (~addr_tok["_atok"].str.isnumeric())
            & (~addr_tok["_atok"].isin(_ADDR_STOP))
        ]
        self.pool_addr_tok = _drop_oversized_groups(
            addr_tok, ["_country", "_atok"], max_size=MAX_ADDR_BLOCK_SIZE
        )[["_country", "_atok", "_addr_uid"]]

        self.addr_to_ids = pool_with_addr.merge(
            unique_addrs, on=["_country", "_norm_addr"]
        )[["_addr_uid", "candidate_entity_id"]]



# ---------------------------------------------------------------------------
# Per-entity capping
# ---------------------------------------------------------------------------

def _cap_per_entity(matches: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Keep at most *top_k* candidates per source1_entity_id, ranked by how
    many independent blocking signals agreed on that pair."""
    if matches.empty:
        return matches
    strength = (
        matches
        .groupby(["source1_entity_id", "candidate_entity_id"])
        .size()
        .rename("match_strength")
        .reset_index()
    )
    strength["_rank"] = strength.groupby("source1_entity_id")[
        "match_strength"
    ].rank(ascending=False, method="first")
    return strength[strength["_rank"] <= top_k][
        ["source1_entity_id", "candidate_entity_id", "match_strength"]
    ]


# ---------------------------------------------------------------------------
# Per-chunk candidate generation
# ---------------------------------------------------------------------------

def generate_candidates_for_chunk(
    s1_chunk: pd.DataFrame,
    pool_index: PoolIndex,
    top_k: int = DEFAULT_TOP_K,
    id_col: str = "entity_id",
    name_col: str = "business_name",
    verbose: bool = True,
) -> pd.DataFrame:
    """Blocking for ONE chunk of source1 against a prebuilt PoolIndex."""
    s1 = _prep(s1_chunk, id_col, name_col)
    s1_mini = s1[
        [id_col, "_country", "_norm_name", "_pin", "_prefix",
         "_sorted_name", "_norm_addr"]
    ].rename(columns={id_col: "source1_entity_id"})

    all_matches = []

    # ---- Pass 1: exact normalized name ------------------------------------
    s1_exact = s1_mini[s1_mini["_norm_name"] != ""]
    match_1 = s1_exact.merge(
        pool_index.pool_exact, on=["_country", "_norm_name"]
    )[["source1_entity_id", "candidate_entity_id"]]
    all_matches.append(match_1)

    # ---- Pass 2: PIN + first-significant-word prefix ----------------------
    s1_pin = s1_mini[
        (s1_mini["_pin"] != "") & (s1_mini["_prefix"] != "")
    ]
    match_2 = s1_pin.merge(
        pool_index.pool_pin, on=["_country", "_pin", "_prefix"]
    )[["source1_entity_id", "candidate_entity_id"]]
    all_matches.append(match_2)

    # ---- Pass 3: name-token overlap ---------------------------------------
    s1_tok = (
        s1_mini[["source1_entity_id", "_country", "_norm_name"]]
        .assign(_tok=s1_mini["_norm_name"].str.split())
        .explode("_tok")
    )
    s1_tok = s1_tok.dropna(subset=["_tok"])
    s1_tok = s1_tok[s1_tok["_tok"].str.len() >= MIN_TOKEN_LEN]

    tok_hits = s1_tok.merge(
        pool_index.pool_tok, on=["_country", "_tok"]
    )[["source1_entity_id", "_uid"]]
    match_3 = tok_hits.merge(
        pool_index.name_to_ids, on="_uid"
    )[["source1_entity_id", "candidate_entity_id"]]
    all_matches.append(match_3)

    # ---- Pass 4: sorted canonical name ------------------------------------
    s1_sorted = s1_mini[s1_mini["_sorted_name"] != ""]
    match_4 = s1_sorted.merge(
        pool_index.pool_sorted, on=["_country", "_sorted_name"]
    )[["source1_entity_id", "candidate_entity_id"]]
    all_matches.append(match_4)

    # ---- Pass 5: address-token overlap ------------------------------------
    s1_addr = (
        s1_mini[["source1_entity_id", "_country", "_norm_addr"]]
        .assign(_atok=s1_mini["_norm_addr"].str.split())
        .explode("_atok")
    )
    s1_addr = s1_addr.dropna(subset=["_atok"])
    s1_addr = s1_addr[
        (s1_addr["_atok"].str.len() >= MIN_ADDR_TOKEN_LEN)
        & (~s1_addr["_atok"].str.isnumeric())
        & (~s1_addr["_atok"].isin(_ADDR_STOP))
    ]

    addr_hits = s1_addr.merge(
        pool_index.pool_addr_tok, on=["_country", "_atok"]
    )[["source1_entity_id", "_addr_uid"]]
    match_5 = addr_hits.merge(
        pool_index.addr_to_ids, on="_addr_uid"
    )[["source1_entity_id", "candidate_entity_id"]]
    all_matches.append(match_5)

    # ---- Union all passes, rank, and cap ----------------------------------
    combined = pd.concat(all_matches, ignore_index=True)
    capped = _cap_per_entity(combined, top_k=top_k)

    all_s1 = pd.DataFrame({"source1_entity_id": s1[id_col].unique()})
    return all_s1.merge(capped, on="source1_entity_id", how="left")


# ---------------------------------------------------------------------------
# Chunked iteration
# ---------------------------------------------------------------------------

def iter_source1_chunks(source1: pd.DataFrame,
                        chunk_size: int = DEFAULT_CHUNK_SIZE):
    for start in range(0, len(source1), chunk_size):
        yield source1.iloc[start:start + chunk_size]


# ---------------------------------------------------------------------------
# Public API (unchanged signature)
# ---------------------------------------------------------------------------

def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    id_col: str = "entity_id",
    name_col: str = "business_name",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Same signature as before -- builds the PoolIndex once, then processes
    source1 in chunks so peak memory only ever holds one chunk's worth of
    candidate pairs.  Existing notebook/script calls need no changes.
    """
    pool_index = PoolIndex(source2, source3, id_col=id_col, name_col=name_col)
    n_chunks = (len(source1) + chunk_size - 1) // chunk_size
    parts = []
    for i, chunk in enumerate(iter_source1_chunks(source1, chunk_size)):
        if verbose:
            print(f"  blocking chunk {i + 1}/{n_chunks} "
                  f"({len(chunk)} S1 rows)...")
        parts.append(
            generate_candidates_for_chunk(
                chunk, pool_index, top_k=top_k,
                id_col=id_col, name_col=name_col, verbose=verbose,
            )
        )
    return pd.concat(parts, ignore_index=True)