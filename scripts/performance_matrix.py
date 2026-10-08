"""Print (and optionally save) a Markdown performance matrix for the transaction
models on the held-out test set, so README numbers are generated, not typed.

Rows per model: default threshold 0.5, the best-F1 point and the 90%-recall
point on the precision-recall curve (same definitions as checkup_pr.py).
PR-AUC here is average precision.

Run from anywhere:  python scripts/performance_matrix.py [--out docs/performance.md]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from src.api.model_service import default_model_dir
from src.data.preprocess import FEATURE_COLUMNS

MODELS = [("LightGBM", "lightgbm.joblib"), ("Random Forest", "random_forest.joblib")]


def row(name, label, y, p, thr, roc, ap):
    pred = (p >= thr).astype(int)
    return (f"| {name} | {label} (thr {thr:.3f}) | {precision_score(y, pred, zero_division=0):.3f} "
            f"| {recall_score(y, pred):.3f} | {f1_score(y, pred):.3f} | {roc:.3f} | {ap:.3f} |")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--parquet", default=str(ROOT / "outputs/data/test_featured.parquet"))
    ap_.add_argument("--out")
    args = ap_.parse_args()

    df = pd.read_parquet(args.parquet)
    X, y = df[FEATURE_COLUMNS], df["is_fraud"].to_numpy()
    lines = [f"Test set: {len(df):,} transactions, {int(y.sum()):,} fraud ({y.mean():.2%}). "
             "PR-AUC = average precision.\n",
             "| Model | Operating point | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
             "|---|---|---|---|---|---|---|"]

    for name, fname in MODELS:
        path = default_model_dir() / fname
        if not path.exists():
            print(f"skipping {name}: {path} not found", file=sys.stderr)
            continue
        p = joblib.load(path).predict_proba(X)[:, 1]
        roc, ap = roc_auc_score(y, p), average_precision_score(y, p)
        prec, rec, thr = precision_recall_curve(y, p)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        best = thr[f1[:-1].argmax()]
        r90 = thr[np.where(rec[:-1] >= 0.90)[0][-1]]
        lines += [row(name, "default", y, p, 0.5, roc, ap),
                  row(name, "best F1", y, p, best, roc, ap),
                  row(name, "90% recall", y, p, r90, roc, ap)]

    table = "\n".join(lines)
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"\nsaved -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
