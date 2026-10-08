"""Export a small, curated set of featurized test transactions for the
dashboard (JSON). Scores them once with the real LightGBM model ONLY to pick
an honest mix: clear catches, a few misses, a few false alarms, and typical
legitimate traffic. Scores are NOT written out - the dashboard gets them
live from the API.

Run from anywhere:  python scripts/export_dashboard_samples.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.api.model_service import default_model_dir
from src.data.preprocess import FEATURE_COLUMNS

REVIEW_THRESHOLD = 0.747  # keep in sync with FraudEvent._recommend_action
EXTRAS = ["category", "trans_date_trans_time"]  # display-only, if present

ROUND = {"amt": 2, "geo_distance_km": 2, "mins_since_last_txn": 1,
         "amt_zscore": 3, "category_fraud_rate": 5}


def pick(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return pool.sample(n=min(n, len(pool)), random_state=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(ROOT / "outputs/data/test_featured.parquet"))
    ap.add_argument("--out", default=str(ROOT / "outputs/dashboard/sample-transactions.json"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    model = joblib.load(default_model_dir() / "lightgbm.joblib")
    df = df.assign(_score=model.predict_proba(df[FEATURE_COLUMNS])[:, 1])

    fraud, legit = df[df["is_fraud"] == 1], df[df["is_fraud"] == 0]
    groups = {
        "caught fraud":    pick(fraud[fraud["_score"] >= REVIEW_THRESHOLD], 14, args.seed),
        "missed fraud":    pick(fraud[fraud["_score"] < 0.2], 3, args.seed),
        "false alarm":     pick(legit[legit["_score"] >= REVIEW_THRESHOLD], 3, args.seed),
        "typical legit":   pick(legit[legit["_score"] < 0.05], 40, args.seed),
    }
    for name, g in groups.items():
        print(f"{name:14s} {len(g):3d} rows")

    chosen = pd.concat(groups.values()).sample(frac=1, random_state=args.seed)
    keep_extras = [c for c in EXTRAS if c in chosen.columns]

    records = []
    for i, (_, row) in enumerate(chosen.iterrows(), start=1):
        feats = {}
        for c in FEATURE_COLUMNS:
            v = row[c]
            feats[c] = int(v) if c in ("hour", "day_of_week", "city_pop") else round(float(v), ROUND.get(c, 4))
        rec = {"id": f"txn_{i:03d}", "features": feats, "label_is_fraud": int(row["is_fraud"])}
        for c in keep_extras:
            rec[c] = str(row[c])
        records.append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} transactions -> {out}")


if __name__ == "__main__":
    main()