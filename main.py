import json
import math
from typing import Any

from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

# Import the FastAPI app built in your service core
from app_core import app as core_app


# -------- Strict, RFC-compliant JSON for every route --------
def _sanitize(obj: Any) -> Any:
    """
    Convert any non-finite numbers to None and normalize numpy/pandas scalars
    so json.dumps(..., allow_nan=False) always succeeds.
    """
    # Optional imports guarded so we don't hard-depend at import time
    try:
        import numpy as np
        NP_FLOAT = (np.floating,)
        NP_INT = (np.integer,)
    except Exception:
        NP_FLOAT = tuple()
        NP_INT = tuple()

    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if NP_FLOAT and isinstance(obj, NP_FLOAT):
        v = float(obj)
        return v if math.isfinite(v) else None

    if NP_INT and isinstance(obj, NP_INT):
        return int(obj)

    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(v) for v in obj]

    try:
        import pandas as pd
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


class SafeJSONResponse(JSONResponse):
    """
    Response class that emits strict RFC-8259 JSON:
      - NaN / +Inf / -Inf -> null
      - allow_nan=False (no invalid tokens)
    """
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(_sanitize(content), ensure_ascii=False, allow_nan=False).encode("utf-8")


# Adopt the core app
app = core_app

# CORS for swagger/curl/GPT/etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Force SafeJSONResponse on ALL currently-registered API routes
for route in list(app.routes):
    if isinstance(route, APIRoute):
        route.response_class = SafeJSONResponse


# JSON error handlers so failures are still valid JSON (jq-parsable)
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return SafeJSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "status": exc.status_code, "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    # Avoid leaking internals in prod; keep it terse but JSON.
    return SafeJSONResponse(
        status_code=500,
        content={"error": "internal_error", "status": 500, "detail": "Unhandled server error"},
    )


# Small probe
@app.get("/info")
def info():
    return {
        "ok": True,
        "title": getattr(app, "title", "Ultimate Quest Service (Small-Payload API)"),
        "version": getattr(app, "version", "unknown"),
        "default_response_class": "SafeJSONResponse",
    }
