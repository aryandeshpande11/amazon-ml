# features.py
"""
Pairwise feature computation for (Source 1 entity, candidate) pairs.
Used identically at training time (on labeled candidate pairs) and at
inference time (on candidate_pairs.tsv rows).

Fix in this version: NaN in business_address/business_name/country was
sometimes surviving as a raw Python float after .astype(str), depending
on the column's underlying dtype (nullable "string" dtype stores missing
values as pd.NA, not the string "nan", and .apply() does not skip NA the
way .str accessor methods do). Every raw text column is now explicitly
.fillna("") BEFORE any string op, and the two .apply() lambdas are
defensive against non-string input as a second safety net.
"""
import re
import pandas as pd
from rapidfuzz import fuzz
from preprocessing import normalize_name

FEATURE_COLUMNS = [
    # Name similarities
    "name_exact", "name_levenshtein", "name_jaro_winkler",
    "name_token_sort", "name_token_set", "name_jaccard",
    "first_token_match", "token_count_diff", "char_len_diff", "name_len_ratio",
    # Address similarities & flags
    "addr_exact", "addr_levenshtein", "addr_token_sort", "addr_token_set", "addr_jaccard",
    "addr_len_ratio", "addr_missing_cand",
    # Postal code & numbers
    "pin_match", "pin_known_both",
    "numeric_overlap_ratio", "num_exact_match", "num_disjoint",
    # Location & landmarks
    "landmark_overlap", "country_match", "country_known_both",
    # Provenance & blocking signals
    "source_is_s2", "match_strength",
]

_PIN_RE = re.compile(r"\b(\d{5,6})\b")
_NUM_RE = re.compile(r"\d+")


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _numeric_overlap(nums_a: set, nums_b: set) -> float:
    if not nums_a or not nums_b:
        return 0.0
    return len(nums_a & nums_b) / max(len(nums_a | nums_b), 1)


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if max(la, lb) == 0:
        return 1.0
    return min(la, lb) / max(la, lb)


def _safe_findall(s) -> set:
    """Defensive against any non-string surviving fillna (e.g. pd.NaT, weird dtypes)."""
    if not isinstance(s, str):
        return set()
    return set(_NUM_RE.findall(s))


def _safe_landmark(s) -> str:
    if not isinstance(s, str):
        return ""
    for kw in ("near", "opposite", "behind"):
        idx = s.find(kw)
        if idx != -1:
            return s[idx:idx + 40].strip()
    return ""


def _build_cache(all_ids, records: pd.DataFrame) -> dict:
    """
    Vectorized replacement for the old per-id Python loop. One subset +
    one set of vectorized string ops over however many unique entities
    are actually referenced in this batch/pairs table.
    """
    sub = records.loc[all_ids].copy()

    name_raw = sub.get("business_name")
    if name_raw is None:
        name_raw = pd.Series("", index=sub.index)
    # Apply the same normalization used in blocking (legal suffix + stopword removal)
    norm_name = name_raw.fillna("").apply(normalize_name)

    addr_raw = sub.get("business_address")
    if addr_raw is None:
        addr_raw = pd.Series("", index=sub.index)
    addr_raw_str = addr_raw.fillna("").astype(str).str.lower()
    addr_missing = addr_raw.isna() | addr_raw_str.isin(["", "nan", "none"])

    pin = addr_raw_str.str.extract(_PIN_RE, expand=False).fillna("")
    numbers = addr_raw_str.apply(_safe_findall)
    landmark = addr_raw_str.apply(_safe_landmark)
    norm_addr = addr_raw_str.str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

    country = sub.get("country")
    if country is None:
        country = pd.Series("", index=sub.index)
    country = country.fillna("").astype(str).str.strip().str.lower()

    first_tok = norm_name.apply(lambda s: s.split()[0] if s else "")

    cache = {}
    for eid, n, ad, p, nu, lm, c, ft, am in zip(
        sub.index, norm_name.values, norm_addr.values, pin.values,
        numbers.values, landmark.values, country.values, first_tok.values, addr_missing.values
    ):
        cache[eid] = {
            "name": n,
            "addr": ad,
            "pin": p,
            "numbers": nu,
            "landmark": lm,
            "country": c,
            "first_tok": ft,
            "addr_missing": am,
        }
    return cache


def _features_for_pairs(pairs: pd.DataFrame, cache: dict) -> pd.DataFrame:
    feats = []
    has_strength = "match_strength" in pairs.columns
    strengths = pairs["match_strength"].fillna(1.0).values if has_strength else None

    for idx, (s1_id, cand_id) in enumerate(zip(pairs["source1_entity_id"], pairs["candidate_entity_id"])):
        a, b = cache[s1_id], cache[cand_id]
        pin_known_both = bool(a["pin"]) and bool(b["pin"])
        country_known_both = bool(a["country"]) and bool(b["country"])

        has_nums_a = bool(a["numbers"])
        has_nums_b = bool(b["numbers"])
        both_have_nums = has_nums_a and has_nums_b

        num_exact = float(both_have_nums and a["numbers"] == b["numbers"])
        num_disjoint = float(both_have_nums and not bool(a["numbers"] & b["numbers"]))

        a_tokens = a["name"].split()
        b_tokens = b["name"].split()

        strength_val = float(strengths[idx]) if strengths is not None else 1.0

        feats.append({
            "name_exact": float(a["name"] == b["name"] and a["name"] != ""),
            "name_levenshtein": fuzz.ratio(a["name"], b["name"]) / 100.0,
            "name_jaro_winkler": fuzz.WRatio(a["name"], b["name"]) / 100.0,
            "name_token_sort": fuzz.token_sort_ratio(a["name"], b["name"]) / 100.0,
            "name_token_set": fuzz.token_set_ratio(a["name"], b["name"]) / 100.0,
            "name_jaccard": _jaccard(a["name"], b["name"]),
            "first_token_match": float(bool(a["first_tok"]) and a["first_tok"] == b["first_tok"]),
            "token_count_diff": float(abs(len(a_tokens) - len(b_tokens))),
            "char_len_diff": float(abs(len(a["name"]) - len(b["name"]))),
            "name_len_ratio": _len_ratio(a["name"], b["name"]),

            "addr_exact": float(a["addr"] == b["addr"] and a["addr"] != ""),
            "addr_levenshtein": fuzz.ratio(a["addr"], b["addr"]) / 100.0,
            "addr_token_sort": fuzz.token_sort_ratio(a["addr"], b["addr"]) / 100.0,
            "addr_token_set": fuzz.token_set_ratio(a["addr"], b["addr"]) / 100.0,
            "addr_jaccard": _jaccard(a["addr"], b["addr"]),
            "addr_len_ratio": _len_ratio(a["addr"], b["addr"]),
            "addr_missing_cand": float(b["addr_missing"]),

            "pin_match": float(pin_known_both and a["pin"] == b["pin"]),
            "pin_known_both": float(pin_known_both),
            "numeric_overlap_ratio": _numeric_overlap(a["numbers"], b["numbers"]),
            "num_exact_match": num_exact,
            "num_disjoint": num_disjoint,

            "landmark_overlap": fuzz.token_set_ratio(a["landmark"], b["landmark"]) / 100.0
                                  if a["landmark"] and b["landmark"] else 0.0,
            "country_match": float(country_known_both and a["country"] == b["country"]),
            "country_known_both": float(country_known_both),

            "source_is_s2": float(str(cand_id).startswith("S2-")),
            "match_strength": strength_val,
        })
    return pd.DataFrame(feats, index=pairs.index)


def _add_candidate_context(out: pd.DataFrame) -> pd.DataFrame:
    strength_component = (out["match_strength"] / 5.0).clip(0.0, 1.0) * 0.1 if "match_strength" in out.columns else 0.0
    out["_rank_score"] = (
        out["name_token_set"] * 0.4
        + out["addr_token_set"] * 0.3
        + out["name_jaccard"] * 0.2
        + strength_component
    )
    out["candidate_rank"] = out.groupby("source1_entity_id")["_rank_score"].rank(ascending=False, method="first")
    out["n_competitors"] = out.groupby("source1_entity_id")["_rank_score"].transform("count")

    def _margin(g):
        sorted_scores = g.sort_values(ascending=False).values
        if len(sorted_scores) < 2:
            return pd.Series([1.0] * len(g), index=g.index)
        top, second = sorted_scores[0], sorted_scores[1]
        return pd.Series([top - second if v == top else v - top for v in g], index=g.index)

    out["score_margin"] = out.groupby("source1_entity_id")["_rank_score"].transform(_margin)
    return out.drop(columns=["_rank_score"])


def build_pair_features(pairs: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.dropna(subset=["candidate_entity_id"]).copy()
    all_ids = pd.unique(pd.concat([pairs["source1_entity_id"], pairs["candidate_entity_id"]]))
    cache = _build_cache(all_ids, records)
    feat_df = _features_for_pairs(pairs, cache)

    # Avoid duplicate column names if pairs already had match_strength
    keep_cols = [c for c in pairs.columns if c in ["source1_entity_id", "candidate_entity_id", "label"]]
    out = pd.concat([pairs[keep_cols], feat_df], axis=1)
    return _add_candidate_context(out)


def build_pair_features_batches(pairs: pd.DataFrame, records: pd.DataFrame, batch_size: int = 300_000,
                                 verbose: bool = True):
    pairs = pairs.dropna(subset=["candidate_entity_id"]).reset_index(drop=True)
    n_batches = (len(pairs) + batch_size - 1) // batch_size
    for i in range(n_batches):
        batch = pairs.iloc[i * batch_size:(i + 1) * batch_size]
        if verbose:
            print(f"  features batch {i + 1}/{n_batches} ({len(batch)} pairs)...")
        all_ids = pd.unique(pd.concat([batch["source1_entity_id"], batch["candidate_entity_id"]]))
        cache = _build_cache(all_ids, records)
        feat_df = _features_for_pairs(batch, cache)
        keep_cols = [c for c in batch.columns if c in ["source1_entity_id", "candidate_entity_id", "label"]]
        out = pd.concat([batch[keep_cols], feat_df], axis=1)
        yield _add_candidate_context(out)


ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + ["candidate_rank", "n_competitors", "score_margin"]