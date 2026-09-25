# features.py
"""
Pairwise feature computation for (Source 1 entity, candidate) pairs.
Used identically at training time (on labeled candidate pairs) and at
inference time (on candidate_pairs.tsv rows).
"""
import pandas as pd
from rapidfuzz import fuzz
from preprocessing import normalize_name, normalize_address

FEATURE_COLUMNS = [
    "name_exact", "name_levenshtein", "name_jaro_winkler",
    "name_token_sort", "name_token_set", "name_jaccard",
    "addr_levenshtein", "addr_token_sort", "addr_token_set",
    "pin_match", "pin_known_both", "numeric_overlap_ratio",
    "landmark_overlap", "country_match", "country_known_both",
    "name_len_ratio", "addr_len_ratio", "source_is_s2",
]


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


def build_pair_features(pairs: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """
    pairs   : DataFrame with columns source1_entity_id, candidate_entity_id
    records : DataFrame indexed by entity_id with business_name,
              business_address, country for ALL sources (S1 + S2 + S3),
              i.e. pd.concat([source1, source2, source3]).set_index("entity_id")

    Returns pairs with FEATURE_COLUMNS appended (rows with a null
    candidate_entity_id, i.e. no-candidate S1 entities, are dropped here --
    handle those as automatic singletons upstream).
    """
    pairs = pairs.dropna(subset=["candidate_entity_id"]).copy()

    # cache normalized text per unique entity to avoid recomputation
    all_ids = pd.unique(pd.concat([pairs["source1_entity_id"], pairs["candidate_entity_id"]]))
    cache = {}
    for eid in all_ids:
        row = records.loc[eid]
        addr = normalize_address(row.get("business_address"))
        cache[eid] = {
            "name": normalize_name(row.get("business_name")),
            "addr": addr["norm"],
            "pin": addr["pin"],
            "numbers": addr["numbers"],
            "landmark": addr["landmark"],
            "country": str(row.get("country") or "").strip().lower(),
        }

    feats = []
    for s1_id, cand_id in zip(pairs["source1_entity_id"], pairs["candidate_entity_id"]):
        a, b = cache[s1_id], cache[cand_id]

        pin_known_both = bool(a["pin"]) and bool(b["pin"])
        country_known_both = bool(a["country"]) and bool(b["country"])

        feats.append({
            "name_exact": float(a["name"] == b["name"] and a["name"] != ""),
            "name_levenshtein": fuzz.ratio(a["name"], b["name"]) / 100.0,
            "name_jaro_winkler": fuzz.WRatio(a["name"], b["name"]) / 100.0,
            "name_token_sort": fuzz.token_sort_ratio(a["name"], b["name"]) / 100.0,
            "name_token_set": fuzz.token_set_ratio(a["name"], b["name"]) / 100.0,
            "name_jaccard": _jaccard(a["name"], b["name"]),
            "addr_levenshtein": fuzz.ratio(a["addr"], b["addr"]) / 100.0,
            "addr_token_sort": fuzz.token_sort_ratio(a["addr"], b["addr"]) / 100.0,
            "addr_token_set": fuzz.token_set_ratio(a["addr"], b["addr"]) / 100.0,
            "pin_match": float(pin_known_both and a["pin"] == b["pin"]),
            "pin_known_both": float(pin_known_both),
            "numeric_overlap_ratio": _numeric_overlap(a["numbers"], b["numbers"]),
            "landmark_overlap": fuzz.token_set_ratio(a["landmark"], b["landmark"]) / 100.0
                                  if a["landmark"] and b["landmark"] else 0.0,
            "country_match": float(country_known_both and a["country"] == b["country"]),
            "country_known_both": float(country_known_both),
            "name_len_ratio": _len_ratio(a["name"], b["name"]),
            "addr_len_ratio": _len_ratio(a["addr"], b["addr"]),
            "source_is_s2": float(str(cand_id).startswith("S2-")),
        })

    feat_df = pd.DataFrame(feats, index=pairs.index)
    out = pd.concat([pairs, feat_df], axis=1)

    # candidate-context features: rank and margin among competitors for the
    # same S1 entity, using name_token_set as the ranking score
    out["_rank_score"] = out["name_token_set"] * 0.5 + out["addr_token_set"] * 0.3 + out["name_jaccard"] * 0.2
    out["candidate_rank"] = out.groupby("source1_entity_id")["_rank_score"].rank(ascending=False, method="first")
    out["n_competitors"] = out.groupby("source1_entity_id")["_rank_score"].transform("count")

    def _margin(g):
        sorted_scores = g.sort_values(ascending=False).values
        if len(sorted_scores) < 2:
            return pd.Series([1.0] * len(g), index=g.index)
        top, second = sorted_scores[0], sorted_scores[1]
        return pd.Series(
            [top - second if v == top else v - top for v in g],
            index=g.index,
        )

    out["score_margin"] = out.groupby("source1_entity_id")["_rank_score"].transform(_margin)
    out.drop(columns=["_rank_score"], inplace=True)

    return out


ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + ["candidate_rank", "n_competitors", "score_margin"]