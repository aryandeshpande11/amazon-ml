# predict.py
"""
Inference pipeline: blocking -> features -> scoring -> thresholding ->
writes candidate_pairs.tsv and matching_results.tsv in the required format.

Usage:
    python predict.py --data-dir dataset/test --model model.json --out-dir output
"""
import argparse
import json
import pandas as pd
import xgboost as xgb

from blocking import generate_candidates
from features import build_pair_features, ALL_FEATURE_COLUMNS


def main(args):
    source1 = pd.read_csv(f"{args.data_dir}/test_source1.tsv", sep="\t", dtype=str)
    source2 = pd.read_csv(f"{args.data_dir}/test_source2.tsv", sep="\t", dtype=str)
    source3 = pd.read_csv(f"{args.data_dir}/test_source3.tsv", sep="\t", dtype=str)

    print("Generating candidates (blocking)...")
    candidates = generate_candidates(source1, source2, source3, top_k=args.top_k)

    # candidate_pairs.tsv -- write BEFORE any model filtering, exactly what
    # gets fed to the matcher
    cand_out = (
        candidates.dropna(subset=["candidate_entity_id"])
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda s: ",".join(s))
        .reindex(source1["entity_id"].rename("source1_entity_id"), fill_value="")
        .reset_index()
        .rename(columns={"candidate_entity_id": "candidate_entity_ids"})
    )
    cand_out.to_csv(f"{args.out_dir}/candidate_pairs.tsv", sep="\t", index=False)

    records = pd.concat([source1, source2, source3], ignore_index=True).set_index("entity_id")
    print("Building features...")
    feat = build_pair_features(candidates, records)

    model = xgb.XGBClassifier()
    model.load_model(args.model)
    with open(args.model + ".meta.json") as f:
        meta = json.load(f)

    feat["score"] = model.predict_proba(feat[ALL_FEATURE_COLUMNS])[:, 1]

    keep = feat[feat["score"] >= meta["threshold"]]
    if meta.get("margin_min", 0) > 0:
        keep = keep[keep["score_margin"] >= meta["margin_min"]]

    match_out = (
        keep.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda s: ",".join(sorted(set(s))))
        .reindex(source1["entity_id"].rename("source1_entity_id"), fill_value="")
        .reset_index()
        .rename(columns={"candidate_entity_id": "matched_entity_ids"})
    )
    match_out.to_csv(f"{args.out_dir}/matching_results.tsv", sep="\t", index=False)
    print(f"Wrote {args.out_dir}/candidate_pairs.tsv and {args.out_dir}/matching_results.tsv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="dataset/test")
    p.add_argument("--model", default="model.json")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--top-k", type=int, default=30)
    main(p.parse_args())