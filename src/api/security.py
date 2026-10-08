"""API-key check, applied per-route with Depends(require_api_key)."""
import hmac
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

# auto_error=False -> we raise our own 401. Declaring the scheme is also what
# gives /docs its "Authorize" button.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Security(_api_key_header)) -> None:
    expected = os.environ.get("FRAUD_API_KEY")
    if not expected:
        # Fail closed: never run unprotected because a secret is missing.
        raise HTTPException(status_code=500, detail="Server misconfigured: FRAUD_API_KEY is not set")
    # compare_digest = constant-time compare (no timing side-channel).
    if provided is None or not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")