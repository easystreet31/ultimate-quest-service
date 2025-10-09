import json, math
from typing import Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

# Import the app created in your service core
from app_core import app as core_app

def _sanitize(obj: Any) -> Any:
    """Convert NaN/+Inf/−Inf → None; normalize numpy/pandas scalars."""
    try:
        import numpy as np
        NP_FLOAT = (np.floating,)
        NP_INT = (np.integer,)
    except Exception:
        NP_FLOAT = tuple(); NP_INT = tuple()

    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if NP_FLOAT and isinstance(obj, NP_FLOAT):
        v = float(obj); return v if math.isfinite(v) else None
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
    """RFC‑compliant JSON (no NaN/Inf) for every route."""
    media_type = "application/json"
    def render(self, content: Any) -> bytes:
        return json.dumps(_sanitize(content), ensure_ascii=False, allow_nan=False).encode("utf-8")

# Adopt the core app
app = core_app

# CORS for swagger/curl/GPT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Force SafeJSONResponse on all API routes
for route in list(app.routes):
    if isinstance(route, APIRoute):
        route.response_class = SafeJSONResponse

@app.get("/info")
def info():
    return {
        "ok": True,
        "title": getattr(app, "title", "Quest Service"),
        "version": getattr(app, "version", "unknown"),
        "default_response_class": "SafeJSONResponse",
    }
