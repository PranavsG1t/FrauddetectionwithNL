import time 
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score, #f1 score is tells you how good is your recall and precision
    roc_auc_score,
    confusion_matrix,
)
import pandas as pd
import lightgbm as lgb
import shap
import numpy as np
import joblib
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from data.preprocess import FEATURE_COLUMNS

DATA_DIR = Path(__file__).resolve().parents[1] / "outputs" / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "outputs" / "models"
MODEL_DIR.mkdir(parents= True, exist_ok=True)

RANDOM_STATE = 42
SHAP_SAMPLE_SIZE = 5000 #full test set is too slow for SHAP, sample for explainer

def load_features():
    train = pd.read_parquet(DATA_DIR/ "train_featured.parquet")
    test = pd.read_parquet(DATA_DIR/ "test_featured.parquet")
    return train, test

def split(df: pd.DataFrame):
    """
    Docstring for split
    
    :param df: Description
    :type df: pd.DataFrame
    """
    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]
    return X,y

def evaluate(name: str, y_true, y_pred, y_proba, fit_seconds: float) -> dict: 
    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
    return {
        "model": name,
        "precision" : precision_score(y_true, y_pred, zero_division=0),
        "recall" : recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "train_seconds": fit_seconds,
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

def train_random_forest(X_train, y_train):
    print("\n Training Random Forest (class_wieght=='balanced')...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=300,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state= RANDOM_STATE,
    )
    start = time.time()
    rf.fit(X_train,y_train)
    fit_seconds = time.time() - start
    print(f"Rf fit [fit_seconds: 1f]s")
    return rf, fit_seconds

def train_lightgbm(X_train,y_train):
    print("\nTraining Lightgbm (is_unbalance=True)...")
    gbm = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        is_unbalance = True,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    start = time.time()
    gbm.fit(X_train, y_train)
    fit_seconds = time.time() - start
    print(f"LightGBM fit in {fit_seconds:.1f}s")
    return gbm, fit_seconds

def runshap(gbm, X_test: pd.DataFrame):
    print(f"\nRunning SHAP(TreeExplainer)on a sample of{SHAP_SAMPLE_SIZE} test rows..")
    sample = X_test.sample(
        n=min(SHAP_SAMPLE_SIZE, len(X_test)), random_state = RANDOM_STATE
    )
    explainer = shap.TreeExplainer(gbm)
    shap_values =explainer.shap_values(sample)
    return explainer, shap_values, sample

def main():
    print("Loading day 1 feature outputs...")
    train, test = load_features()
    X_train, y_train = split(train)
    X_test, y_test = split(test)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features used:{FEATURE_COLUMNS}")
    rf, rf_seconds = train_random_forest(X_train, y_train)
    gbm, gbm_seconds = train_lightgbm(X_train, y_train)
    
    results = []
    for name, model, fit_seconds in [
        ("random_forest",rf, rf_seconds),
        ("lightgbm", gbm, gbm_seconds),
    ]:
        y_pred = model.predict(X_test)
        y_proba =model.predict_proba(X_test)[:,1]
        results.append(evaluate(name, y_test, y_pred, y_proba, fit_seconds))
    
    comparison = pd.DataFrame(results).set_index("model")
    print("\nModel Comparison {threshold = 0.5}:")
    print(comparison.to_string())

    comparison.to_csv(MODEL_DIR / "model_comparison.csv")
    print(f"\nSaved Comparison to {MODEL_DIR / 'model_comparison.csv'}")

    explainer, shap_values, shap_sample = runshap(gbm, X_test)

    print("\nSaving models+SHAP explainer for pipeline integration")
    joblib.dump(rf, MODEL_DIR / "random_forest.joblib")
    joblib.dump(gbm, MODEL_DIR / "lightgbm.joblib")
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.joblib")
    joblib.dump(
        {"shap_values": shap_values, "sample_index": shap_sample.index},
        MODEL_DIR / "shap_values_sample.joblib",
    )
    print(f"Saved to {MODEL_DIR}")

    print("\nDone. Winner (by ROC-AUC) should be the one you take into Day 4's RAG layer.")


if __name__ == "__main__":
    main()