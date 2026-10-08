"""API-layer request/response models (Pydantic).

FraudEvent (src/schema/fraud_event.py) stays a plain dataclass - FastAPI
serves stdlib dataclasses as response models, so we don't rewrite it.
Everything here is about what the API *accepts* from clients.
"""
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str


class TransactionFeatures(BaseModel):
    """The 8 model-ready features, in the same names as
    src/data/preprocess.py::FEATURE_COLUMNS.

    Feature engineering (haversine distance, per-card history, category
    target encoding) happens upstream; the API is the model-serving layer.
    """
    # extra="forbid": a typo like "amount" is a 422, not silently ignored.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {   # looks routine
                    "amt": 42.5, "geo_distance_km": 12.0, "hour": 14,
                    "day_of_week": 2, "mins_since_last_txn": 640.0,
                    "amt_zscore": 0.1, "category_fraud_rate": 0.004,
                    "city_pop": 150000,
                },
                {   # looks suspicious: big amount, far away, 3am, rare spike
                    "amt": 1250.0, "geo_distance_km": 85.0, "hour": 3,
                    "day_of_week": 5, "mins_since_last_txn": 4.0,
                    "amt_zscore": 6.5, "category_fraud_rate": 0.018,
                    "city_pop": 900,
                },
            ]
        },
    )

    amt: float = Field(ge=0, description="Transaction amount")
    geo_distance_km: float = Field(ge=0, description="Cardholder home to merchant, km")
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6, description="Monday=0")
    mins_since_last_txn: float = Field(ge=0, description="Minutes since this card's previous transaction")
    amt_zscore: float = Field(description="Amount vs this cardholder's own history")
    category_fraud_rate: float = Field(ge=0, le=1, description="Historical fraud rate of the merchant category")
    city_pop: int = Field(ge=0)