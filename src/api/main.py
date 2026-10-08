"""Step 2 - scoring endpoint + Gemini alert, API-key auth, CORS, models loaded once.

Run from the project root:
    uvicorn src.api.main:app --reload --port 8000
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.alert_service import AlertService
from src.api.model_service import ModelService, default_model_dir
from src.api.schemas import HealthResponse, TransactionFeatures
from src.api.security import require_api_key
from src.schema.fraud_event import FraudEvent

load_dotenv()  # local dev: reads .env. Never overrides real environment variables.

PLACEHOLDER_EXPLANATION = "Explanation unavailable: alerts are disabled on this server (ENABLE_ALERTS=0)."


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE at startup: models are loaded here, not per request.
    # Order rule from the blueprint: models first, Gemini/gRPC client last.
    app.state.model_service = ModelService.load(default_model_dir())
    # AlertService.build() is where the Gemini client gets created (lazy import).
    # ENABLE_ALERTS=0 skips it: tests and offline dev run without Gemini.
    app.state.alerts = AlertService.build() if os.environ.get("ENABLE_ALERTS", "1") == "1" else None
    yield
    # (code after `yield` would run at shutdown)


app = FastAPI(title="Fraud Detection API", version="0.3.0", lifespan=lifespan)

# CORS: which *browser origins* may read our responses. Explicit list, not "*".
# Only Content-Type is allowed as a custom header: the API key must never be
# held by browser code - the dashboard's server-side route adds it.
_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


# Plain `def` (not `async def`): scoring is CPU-bound (LightGBM + SHAP) and the
# Gemini call blocks. FastAPI runs `def` handlers in a worker thread, so they
# don't freeze the event loop; an `async def` doing blocking work would.
@app.post(
    "/transactions/score",
    response_model=FraudEvent,
    dependencies=[Depends(require_api_key)],
)
def score_transaction(tx: TransactionFeatures, request: Request) -> FraudEvent:
    service: ModelService = request.app.state.model_service
    risk_score, shap_output = service.score(tx.model_dump())

    alerts: AlertService | None = request.app.state.alerts
    if alerts is None:
        return FraudEvent.from_transaction(
            shap_output=shap_output,
            risk_score=risk_score,
            explanation_text=PLACEHOLDER_EXPLANATION,
        )
    # Blocking Gemini call is fine here: `def` handlers run in a worker thread.
    return alerts.explain_transaction(shap_output, risk_score)