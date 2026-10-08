"""Day 4d - Run the full RAG alert pipeline end to end: retrieve -> generate
-> groundedness-check, on a couple of example cases from Day 2/3 output.
"""
import random
import joblib
import pandas as pd
import subprocess
import json

from src.rag.eval import check_groundedness
from src.rag.alert_generator import format_document_facts, format_transaction_facts

FEATURE_COLUMNS = ["amt","geo_distance_km","hour","day_of_week","mins_since_last_txn","amt_zscore", "category_fraud_rate", "city_pop"]
EXAMPLE_DOCUMENT = {"prediction": "tampered", "confidence":0.93, "gradcam_region":"photo area, top-left"}

def get_random_transaction_case(test_parquet_path: str = "/Users/ggpranav/Documents/FrauddetectionNL/outputs/data/test_featured.parquet"):
    """Pick a random test set row, run it throught lightgbm,, and return shap_output, risk_score) for alert pipeline"""
    model = joblib.load("outputs/models/lightgbm.joblib")
    explainer = joblib.load("outputs/models/shap_explainer.joblib")
    df = pd.read_parquet(test_parquet_path)
    row_idx = random.randrange(len(df))
    print(f"DEBUG: picked row_idx={row_idx}",flush=True)
    row = df.iloc[[row_idx]] #[[row_idx]] need to check why its written like this
    print(f"DEBUG: picked row: \n{row[FEATURE_COLUMNS]}",flush=True)

    risk_score = float(model.predict_proba(row[FEATURE_COLUMNS])[:, 1][0])
    print(f"DEBUG: risk_score={risk_score}", flush= True)

    shap_values = explainer.shap_values(row[FEATURE_COLUMNS])
    print(f"DEBUG: shap_values computed, type= {type(shap_values)}", flush=True)

    # LightGBM binary classification can return either a single array
    # (shap for class 1) or a list/3D array of per-class values depending
    # on SHAP version - normalize to a flat 1D array for class 1.
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    elif shap_values.ndim == 3:
        values = shap_values[0, :,1]
    else:
        values = shap_values[0]

    shap_output = dict(zip(FEATURE_COLUMNS, values))
    actual_label = int(row["is_fraud"].iloc[0])
    print(f"(picked test row {row_idx}, actual is_fraud={actual_label})")
    return shap_output, risk_score

def get_random_document_case():
    result = subprocess.run(
        ["python", "scripts/doc_infer_worker.py"],
        capture_output=True, text=True, check=True,
        cwd="/Users/ggpranav/Documents/FrauddetectionNL",
    )
    verdict = json.loads(result.stdout.strip().splitlines()[-1])
    print(f"DEBUG: picked doc {verdict['path']}, actual label={verdict['actual_label']}, predicted={verdict['prediction']}", flush=True)
    return {"prediction": verdict["prediction"], "confidence": verdict["confidence"]}

def run_case(label: str, facts: str, alert: str):
    result = check_groundedness(facts, alert.explanation_text)
    print(f"\n--- {label} ---")
    print(f"Facts: {facts}")
    print(f"Alert: {alert.explanation_text}")
    print(f"  → {alert.recommended_action} (risk_score={alert.risk_score: .3f})")
    print(f"Groundedness: score={result.score:.2f} flagged={result.flagged} missing={result.missing_terms}")


def main():
    print("DEBUG: getting transaction case", flush=True)
    shap_output, risk_score = get_random_transaction_case()
    txn_facts = format_transaction_facts(shap_output)

    print("DEBUG: getting document case", flush=True)
    cv_verdict = get_random_document_case()
    doc_facts = format_document_facts(cv_verdict)

    print("DEBUG: importing alert_genrator + building chain", flush=True)
    from src.rag.alert_generator import( build_alert_chain,generate_document_alert, generate_transaction_alert) #called inside main rather can loading initially 
    
    print("DEBUG: building chain", flush=True)
    chain = build_alert_chain()

    txn_alert = generate_transaction_alert(shap_output, risk_score= risk_score, chain=chain)
    run_case("Transaction case", txn_facts, txn_alert)
    
    doc_alert = generate_document_alert(cv_verdict, chain=chain)
    run_case("Document case", doc_facts, doc_alert)


if __name__ == "__main__":
    main()
