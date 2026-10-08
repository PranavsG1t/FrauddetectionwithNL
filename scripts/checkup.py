import pandas as pd, joblib, numpy as np
from sklearn.metrics import average_precision_score

tr = pd.read_parquet("outputs/data/train_featured.parquet")
te = pd.read_parquet("outputs/data/test_featured.parquet")
print("train/test shape:", tr.shape, te.shape)
print("fraud rate:", tr["is_fraud"].mean(), te["is_fraud"].mean())   # expect ~0.5-0.6%
print("NaNs in engineered cols:\n", tr.isna().sum()[lambda s: s > 0])
print("columns:", list(tr.columns))     # eyeball for leaks: no ids, names, raw trans_num

print(pd.read_csv("outputs/models/model_comparison.csv"))

model = joblib.load("outputs/models/lightgbm.joblib")
feats = list(model.feature_name_) if hasattr(model, "feature_name_") else None
print("model features:", feats)
if feats:
    p = model.predict_proba(te[feats])[:, 1]
    print("test PR-AUC:", average_precision_score(te["is_fraud"], p))
    print("score range:", p.min(), p.max())