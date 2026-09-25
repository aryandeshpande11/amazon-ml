import subprocess
import pandas as pd


def load_ground_truth(path):
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    ids = gt["source1_entity_id"].values
    matched = gt["matched_entity_ids"].values
    return {s1: (set(m.split(",")) if m else set()) for s1, m in zip(ids, matched)}


def label_pairs(pairs_df, ground_truth, s1_col="source1_entity_id", cand_col="candidate_entity_id"):
    """Vectorized labeling: no iterrows. Maps each pair to 1/0 via a set lookup."""
    gt_lookup = ground_truth  # {s1_id: set(matched_ids)}
    s1_ids = pairs_df[s1_col].values
    cand_ids = pairs_df[cand_col].values
    labels = [
        1 if cand in gt_lookup.get(s1, set()) else 0
        for s1, cand in zip(s1_ids, cand_ids)
    ]
    pairs_df = pairs_df.copy()
    pairs_df["label"] = labels
    return pairs_df


def split_source1_ids(source1, val_frac=0.2, random_state=42):
    """
    Group-wise split on source1_entity_id. Splitting at the pair level would
    put some of an entity's candidates in train and others in val, leaking
    that entity's identity across the split -- split on entity_id instead,
    then filter the pairs table by membership.
    """
    ids = source1["entity_id"].sample(frac=1.0, random_state=random_state).values
    cut = int(len(ids) * (1 - val_frac))
    return set(ids[:cut]), set(ids[cut:])


def detect_cuda():
    """Best-effort GPU check so notebooks don't hardcode use_cuda."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def build_model(use_cuda=False):
    """MIT-licensed gradient boosting model, <8B params trivially satisfied."""
    import xgboost as xgb
    params = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )
    if use_cuda:
        params["device"] = "cuda"   # XGBoost >= 2.0 syntax
    return xgb.XGBClassifier(**params)