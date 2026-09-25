import numpy as np
import pandas as pd

def predictions_from_scored_pairs(val_df, threshold, margin_min=0.0):
    """Turn a scored pair table into {source1_entity_id: set(matched_ids)}."""
    preds = {}
    df = val_df[val_df["score"] >= threshold]
    if margin_min > 0.0:
        # keep a candidate only if it beats the next-best candidate by margin_min
        df = df.sort_values(["source1_entity_id", "score"], ascending=[True, False])
        keep_rows = []
        for s1_id, group in df.groupby("source1_entity_id"):
            scores = group["score"].values
            if len(scores) == 1:
                keep_rows.append(group.iloc[[0]])
                continue
            margins = scores[:-1] - scores[1:]
            cutoff = 0
            for i, m in enumerate(margins):
                if m < margin_min:
                    cutoff = i + 1
                else:
                    break
            keep_rows.append(group.iloc[:max(cutoff, 1)])
        df = pd.concat(keep_rows) if keep_rows else df.iloc[0:0]
    for s1_id, group in df.groupby("source1_entity_id"):
        preds[s1_id] = set(group["source2_or_3_entity_id"].tolist())
    return preds

def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom

def entity_level_report(preds, ground_truth):
    per_entity = {}
    n_perfect = n_zero = 0
    singleton_hits = singleton_total = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = preds.get(s1_id, set())

        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                p = r = f = 1.0
                singleton_hits += 1
            else:
                p = r = f = 0.0
        else:
            tp = len(true_set & pred_set)
            p = tp / len(pred_set) if pred_set else 0.0
            r = tp / len(true_set) if true_set else 0.0
            f = f_beta(p, r, beta=0.5)

        per_entity[s1_id] = (p, r, f)
        if f == 1.0:
            n_perfect += 1
        if f == 0.0:
            n_zero += 1

    n = len(per_entity)
    macro_p = np.mean([v[0] for v in per_entity.values()])
    macro_r = np.mean([v[1] for v in per_entity.values()])
    macro_f05 = np.mean([v[2] for v in per_entity.values()])

    return {
        "n_entities": n,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f05": macro_f05,
        "n_perfect": n_perfect,
        "n_zero": n_zero,
        "singleton_accuracy": singleton_hits / singleton_total if singleton_total else float("nan"),
        "singleton_total": singleton_total,
        "per_entity": per_entity,
    }

def sweep_threshold(val_df, val_gt, thresholds=None, margin_min=0.0):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    results = []
    best_t, best_f05 = 0.5, -1.0
    for t in thresholds:
        preds = predictions_from_scored_pairs(val_df, threshold=t, margin_min=margin_min)
        report = entity_level_report(preds, val_gt)
        results.append((t, report["macro_f05"]))
        if report["macro_f05"] > best_f05:
            best_f05, best_t = report["macro_f05"], t
    return best_t, best_f05, results