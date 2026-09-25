"""
Candidate generation (blocking) for Source 1 entities against the pooled
Source 2 + Source 3 records.

Two independent, cheap passes are unioned per S1 entity:
  1. Token/PIN inverted-index pass  (_token_blocking_pass)
  2. TF-IDF char n-gram top-k pass  (_tfidf_topk_pass)

MAX_BLOCK_SIZE is the fix for a MemoryError seen at full scale: a generic
token (e.g. "enterprises", "traders", "pvt") can appear in tens/hundreds of
thousands of pool records. Unioning that block into every S1 entity that
happens to contain the word blows up memory with no recall benefit -- a
block that large carries essentially no discriminative signal anyway, and
those pairs are still reachable through the TF-IDF pass. So any token or
PIN whose posting list exceeds MAX_BLOCK_SIZE is skipped entirely rather
than unioned in.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

from preprocessing import normalize_name, normalize_address

MAX_BLOCK_SIZE = 1000   # skip token/PIN blocks larger than this
MIN_TOKEN_LEN = 4       # ignore very short tokens (too generic to be useful)


def _prep(df: pd.DataFrame, id_col: str, name_col: str, addr_col: str = "business_address") -> pd.DataFrame:
    out = df.copy()
    out["_norm_name"] = out[name_col].apply(normalize_name)
    addr = out[addr_col].apply(normalize_address) if addr_col in out.columns else None
    out["_norm_addr"] = addr.apply(lambda d: d["norm"]) if addr is not None else ""
    out["_pin"] = addr.apply(lambda d: d["pin"]) if addr is not None else ""
    out["_blend"] = (out["_norm_name"] + " " + out["_norm_addr"]).str.strip()
    return out


def _build_inverted_index(pool: pd.DataFrame, id_col: str):
    """
    Builds token->{ids} and pin->{ids} indexes, DROPPING any block that
    exceeds MAX_BLOCK_SIZE as soon as it's detected -- this bounds peak
    memory instead of building the full oversized set and discarding it
    afterwards.
    """
    token_index = defaultdict(set)
    pin_index = defaultdict(set)
    oversized_tokens = set()
    oversized_pins = set()

    ids = pool[id_col].values
    names = pool["_norm_name"].values
    pins = pool["_pin"].values

    for eid, name, pin in zip(ids, names, pins):
        for tok in set(name.split()):
            if len(tok) < MIN_TOKEN_LEN or tok in oversized_tokens:
                continue
            bucket = token_index[tok]
            bucket.add(eid)
            if len(bucket) > MAX_BLOCK_SIZE:
                oversized_tokens.add(tok)
                del token_index[tok]   # free the memory immediately
        if pin and pin not in oversized_pins:
            bucket = pin_index[pin]
            bucket.add(eid)
            if len(bucket) > MAX_BLOCK_SIZE:
                oversized_pins.add(pin)
                del pin_index[pin]

    if oversized_tokens:
        print(f"[blocking] skipped {len(oversized_tokens)} oversized name tokens (>{MAX_BLOCK_SIZE} postings)")
    if oversized_pins:
        print(f"[blocking] skipped {len(oversized_pins)} oversized PIN blocks (>{MAX_BLOCK_SIZE} postings)")

    return token_index, pin_index


def _token_blocking_pass(source1: pd.DataFrame, pool: pd.DataFrame, id_col: str):
    token_index, pin_index = _build_inverted_index(pool, id_col)

    candidates = defaultdict(set)
    ids = source1[id_col].values
    names = source1["_norm_name"].values
    pins = source1["_pin"].values

    for eid, name, pin in zip(ids, names, pins):
        cand = candidates[eid]
        for tok in set(name.split()):
            if len(tok) < MIN_TOKEN_LEN:
                continue
            bucket = token_index.get(tok)
            if bucket:
                cand |= bucket
        if pin:
            bucket = pin_index.get(pin)
            if bucket:
                cand |= bucket
    return candidates


def _tfidf_topk_pass(source1: pd.DataFrame, pool: pd.DataFrame, k: int, id_col: str):
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=50_000
    )
    pool_vecs = vectorizer.fit_transform(pool["_blend"])
    s1_vecs = vectorizer.transform(source1["_blend"])

    n_neighbors = min(k, len(pool))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(pool_vecs)
    distances, indices = nn.kneighbors(s1_vecs)

    s1_ids = source1[id_col].values
    pool_ids = pool[id_col].values

    candidates = defaultdict(set)
    for row_i in range(len(s1_ids)):
        eid = s1_ids[row_i]
        for dist, pool_idx in zip(distances[row_i], indices[row_i]):
            if 1 - dist > 0:
                candidates[eid].add(pool_ids[pool_idx])
    return candidates


def merge_candidate_passes(*passes) -> dict:
    merged = defaultdict(set)
    for pass_dict in passes:
        for eid, cand_set in pass_dict.items():
            merged[eid] |= cand_set
    return merged


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    top_k: int = 30,
    id_col: str = "entity_id",
    name_col: str = "business_name",
) -> pd.DataFrame:
    """
    Returns a long-format DataFrame: source1_entity_id, candidate_entity_id.
    S1 entities with zero candidates get one row with candidate_entity_id
    = None so they survive downstream as declared singletons rather than
    silently disappearing.
    """
    s1 = _prep(source1, id_col, name_col)
    pool = _prep(pd.concat([source2, source3], ignore_index=True), id_col, name_col)

    token_cands = _token_blocking_pass(s1, pool, id_col=id_col)
    tfidf_cands = _tfidf_topk_pass(s1, pool, k=top_k, id_col=id_col)
    merged = merge_candidate_passes(token_cands, tfidf_cands)

    rows = []
    for eid in s1[id_col].values:
        cand_set = merged.get(eid, set())
        if cand_set:
            for cand_id in cand_set:
                rows.append((eid, cand_id))
        else:
            rows.append((eid, None))

    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])