import pandas as pd, joblib, numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

te = pd.read_parquet("outputs/data/test_featured.parquet")
y = te["is_fraud"]

for name in ["random_forest", "lightgbm"]:
    m = joblib.load(f"outputs/models/{name}.joblib")
    feats = list(m.feature_name_) if hasattr(m, "feature_name_") else list(m.feature_names_in_)
    p = m.predict_proba(te[feats])[:, 1]
    ap = average_precision_score(y, p)
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    i = f1[:-1].argmax()
    print(f"\n{name}: PR-AUC={ap:.3f}  features={feats}")
    print(f"  best-F1 threshold={thr[i]:.3f} -> precision={prec[i]:.3f} recall={rec[i]:.3f}")
    j = np.where(rec[:-1] >= 0.90)[0][-1]
    print(f"  90%-recall threshold={thr[j]:.3f} -> precision={prec[j]:.3f}")