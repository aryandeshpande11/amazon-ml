"""
Candidate generation (blocking) for Source 1 entities against the pooled
Source 2 + Source 3 records.

Fixes vs. the previous version, for scale (S1 ~1.7M, S2 ~4M, S3 ~5M rows):

1. The token index is now built from UNIQUE (country, norm_name) pairs,
   not by exploding every one of the ~9M pool rows into tokens. Business
   datasets at this scale have huge numbers of duplicate names across
   branches/locations, so tokenizing every duplicate separately was most
   of the wasted work.
2. Every blocking pass (exact name, PIN+prefix, token) now drops any
   group bigger than MAX_BLOCK_SIZE *before* the merge, not after --
   a single generic name/word could otherwise blow the merge itself up
   to millions of rows even though the final capped result is small.
3. Candidates are capped per Source 1 entity (top_k) using a cheap
   match-strength count, so memory downstream (features, scoring) is
   bounded by a fixed multiple of len(source1), not by however many
   pool records happened to share a token.
4. A PoolIndex is built ONCE and reused across chunks of source1, so the
   expensive pool-side grouping never runs more than once per program run.
   generate_candidates() has the same signature as before -- it just
   chunks internally now, so no notebook/script changes are required to
   get the speedup.
"""
import numpy as np
import pandas as pd

MAX_BLOCK_SIZE = 2000
MIN_TOKEN_LEN = 4
DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_TOP_K = 40


def _prep(df: pd.DataFrame, id_col: str, name_col: str, addr_col: str = "business_address") -> pd.DataFrame:
    """Vectorized normalization -- no per-row python function calls."""
    out = df.copy()
    temp_name = out[name_col].astype(str).str.lower().str.replace("&", " and ", regex=False)
    out["_norm_name"] = temp_name.str.replace(r"[^\w\s]", " ", regex=True).str.strip().fillna("")
    if addr_col in out.columns:
        out["_pin"] = out[addr_col].astype(str).str.extract(r"\b(\d{5,6})\b", expand=False).fillna("")
    else:
        out["_pin"] = ""
    if "country" in out.columns:
        out["_country"] = out["country"].astype(str).str.strip().str.lower()
    else:
        out["_country"] = ""
    out["_prefix"] = out["_norm_name"].str[:3]
    return out


def _drop_oversized_groups(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Drop rows belonging to any group bigger than MAX_BLOCK_SIZE, so a
    generic name/PIN/token can't blow up the merge that follows."""
    counts = df.groupby(group_cols).size().reset_index(name="_cnt")
    valid = counts[counts["_cnt"] <= MAX_BLOCK_SIZE][group_cols]
    return df.merge(valid, on=group_cols)


class PoolIndex:
    """
    Precomputed blocking structures for the pooled S2+S3 records.
    Build ONCE per run (see generate_candidates) and reuse across every
    source1 chunk -- rebuilding this per chunk is what made blocking slow.
    """

    def __init__(self, source2: pd.DataFrame, source3: pd.DataFrame,
                 id_col: str = "entity_id", name_col: str = "business_name"):
        pool = _prep(pd.concat([source2, source3], ignore_index=True), id_col, name_col)
        self.pool_mini = pool[[id_col, "_country", "_norm_name", "_pin", "_prefix"]].rename(
            columns={id_col: "candidate_entity_id"}
        )

        # --- exact-name lookup (capped: drop names shared by > MAX_BLOCK_SIZE records)
        pool_named = self.pool_mini[self.pool_mini["_norm_name"] != ""]
        self.pool_exact = _drop_oversized_groups(pool_named, ["_country", "_norm_name"])

        # --- PIN + name-prefix lookup, capped the same way
        pool_pin = self.pool_mini[(self.pool_mini["_pin"] != "") & (self.pool_mini["_prefix"] != "")]
        self.pool_pin = _drop_oversized_groups(pool_pin, ["_country", "_pin", "_prefix"])

        # --- token index built from UNIQUE (country, norm_name) pairs only.
        # Tokenize each distinct name once; keep a separate name -> ids map
        # to expand matched names back to real candidate_entity_ids later.
        unique_names = pool_named[["_country", "_norm_name"]].drop_duplicates().reset_index(drop=True)
        unique_names["_uid"] = np.arange(len(unique_names))

        tok = unique_names.assign(_tok=unique_names["_norm_name"].str.split()).explode("_tok")
        tok = tok.dropna(subset=["_tok"])
        tok = tok[tok["_tok"].str.len() >= MIN_TOKEN_LEN]
        self.pool_tok = _drop_oversized_groups(tok, ["_country", "_tok"])[["_country", "_tok", "_uid"]]

        self.name_to_ids = pool_named.merge(
            unique_names, on=["_country", "_norm_name"]
        )[["_uid", "candidate_entity_id"]]


def _cap_per_entity(matches: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Keep at most top_k candidates per source1_entity_id, ranked by how
    many independent blocking signals (exact/pin/token) agreed on that pair."""
    if matches.empty:
        return matches
    strength = (
        matches.groupby(["source1_entity_id", "candidate_entity_id"]).size()
        .rename("match_strength").reset_index()
    )
    strength["_rank"] = strength.groupby("source1_entity_id")["match_strength"].rank(
        ascending=False, method="first"
    )
    return strength[strength["_rank"] <= top_k][["source1_entity_id", "candidate_entity_id"]]


def generate_candidates_for_chunk(s1_chunk: pd.DataFrame, pool_index: PoolIndex, top_k: int = DEFAULT_TOP_K,
                                   id_col: str = "entity_id", name_col: str = "business_name",
                                   verbose: bool = True) -> pd.DataFrame:
    """Blocking for ONE chunk of source1 against a prebuilt PoolIndex."""
    s1 = _prep(s1_chunk, id_col, name_col)
    s1_mini = s1[[id_col, "_country", "_norm_name", "_pin", "_prefix"]].rename(columns={id_col: "source1_entity_id"})

    s1_exact = s1_mini[s1_mini["_norm_name"] != ""]
    match_1 = s1_exact.merge(pool_index.pool_exact, on=["_country", "_norm_name"])[
        ["source1_entity_id", "candidate_entity_id"]]

    s1_pin = s1_mini[(s1_mini["_pin"] != "") & (s1_mini["_prefix"] != "")]
    match_2 = s1_pin.merge(pool_index.pool_pin, on=["_country", "_pin", "_prefix"])[
        ["source1_entity_id", "candidate_entity_id"]]

    s1_tok = s1_mini[["source1_entity_id", "_country", "_norm_name"]].assign(
        _tok=s1_mini["_norm_name"].str.split()
    ).explode("_tok")
    s1_tok = s1_tok.dropna(subset=["_tok"])
    s1_tok = s1_tok[s1_tok["_tok"].str.len() >= MIN_TOKEN_LEN]

    tok_hits = s1_tok.merge(pool_index.pool_tok, on=["_country", "_tok"])[["source1_entity_id", "_uid"]]
    match_3 = tok_hits.merge(pool_index.name_to_ids, on="_uid")[["source1_entity_id", "candidate_entity_id"]]

    all_matches = pd.concat([match_1, match_2, match_3], ignore_index=True)
    capped = _cap_per_entity(all_matches, top_k=top_k)

    all_s1 = pd.DataFrame({"source1_entity_id": s1[id_col].unique()})
    return all_s1.merge(capped, on="source1_entity_id", how="left")


def iter_source1_chunks(source1: pd.DataFrame, chunk_size: int = DEFAULT_CHUNK_SIZE):
    for start in range(0, len(source1), chunk_size):
        yield source1.iloc[start:start + chunk_size]


def generate_candidates(source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame,
                         top_k: int = DEFAULT_TOP_K, id_col: str = "entity_id", name_col: str = "business_name",
                         chunk_size: int = DEFAULT_CHUNK_SIZE, verbose: bool = True) -> pd.DataFrame:
    """
    Same signature as before -- builds the PoolIndex once, then processes
    source1 in chunks so peak memory only ever holds one chunk's worth of
    candidate pairs. Existing notebook/script calls need no changes.
    """
    pool_index = PoolIndex(source2, source3, id_col=id_col, name_col=name_col)
    n_chunks = (len(source1) + chunk_size - 1) // chunk_size
    parts = []
    for i, chunk in enumerate(iter_source1_chunks(source1, chunk_size)):
        if verbose:
            print(f"  blocking chunk {i + 1}/{n_chunks} ({len(chunk)} S1 rows)...")
        parts.append(generate_candidates_for_chunk(chunk, pool_index, top_k=top_k, id_col=id_col, name_col=name_col,
                                                     verbose=verbose))
    return pd.concat(parts, ignore_index=True)