import json
import math
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

# Import the FastAPI app and helpers from your service core
from app_core import (
    app as core_app,
    # helpers we use in the probe (all exist in your app_core)
    _pick_url, fetch_xlsx, normalize_leaderboard,
    holdings_from_urls, _load_player_tags, compute_family_qp,
    FamilyEvaluateTradeReq,
)


# ---------- Strict, RFC-compliant JSON for every route ----------

def _sanitize(obj: Any) -> Any:
    """Convert non-finite numbers to None; normalize numpy/pandas scalars."""
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
    """Response class that emits strict RFC-8259 JSON (no NaN/Inf)."""
    media_type = "application/json"
    def render(self, content: Any) -> bytes:
        return json.dumps(_sanitize(content), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _exc_dict(exc: Exception) -> Dict[str, Any]:
    t = type(exc).__name__
    msg = str(exc)
    return {"type": t, "message": msg}


# ---------- Adopt core app & set global behavior ----------

app = core_app

# CORS for swagger/curl/GPT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Force SafeJSONResponse on all current routes
for route in list(app.routes):
    if isinstance(route, APIRoute):
        route.response_class = SafeJSONResponse


# ---------- JSON error handlers (so failures are jq-parsable) ----------

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    # Keep the original status code; include structured detail
    base = {"error": "http_error", "status": exc.status_code}
    # If detail is plain str, wrap it
    if isinstance(exc.detail, str):
        base["detail"] = exc.detail
    else:
        base["detail"] = exc.detail
    return SafeJSONResponse(status_code=exc.status_code, content=base)

@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    # Avoid HTML – return structured JSON with error type/message
    return SafeJSONResponse(
        status_code=500,
        content={"error": "internal_error", "status": 500, "detail": _exc_dict(exc)},
    )


# ---------- Probe route to isolate the failing step ----------

@app.post("/__probe_evaluate")
def probe_family_evaluate(req: Dict[str, Any]):
    """
    Lightweight diagnostic that runs the same *inputs pipeline* as
    /family_evaluate_trade_by_urls, step-by-step, and reports where it fails.
    """
    steps: List[Dict[str, Any]] = []
    prefer = bool(req.get("prefer_env_defaults", True))

    def _add(step: str, **kw):
        d = {"step": step, **kw}
        steps.append(d)
        return d

    # 1) Resolve leaderboard URL
    try:
        url_lb = _pick_url(req.get("leaderboard_url"), "leaderboard", prefer)
        _add("pick_leaderboard_url", ok=True, url=url_lb)
    except Exception as e:
        _add("pick_leaderboard_url", ok=False, error=_exc_dict(e))
        return {"ok": False, "steps": steps}

    # 2) Fetch leaderboard XLSX
    try:
        lb_sheets = fetch_xlsx(url_lb)
        _add("fetch_leaderboard_xlsx", ok=True, sheet_count=len(lb_sheets), sheet_names=list(lb_sheets.keys())[:5])
    except Exception as e:
        _add("fetch_leaderboard_xlsx", ok=False, error=_exc_dict(e))
        return {"ok": False, "steps": steps}

    # 3) Normalize leaderboard
    try:
        leader = normalize_leaderboard(lb_sheets)
        _add("normalize_leaderboard", ok=True, players=len(leader))
    except Exception as e:
        _add("normalize_leaderboard", ok=False, error=_exc_dict(e))
        return {"ok": False, "steps": steps}

    # 4) Load holdings for all family accounts
    try:
        accounts_before = holdings_from_urls(
            req.get("holdings_e31_url"),
            req.get("holdings_dc_url"),
            req.get("holdings_fe_url"),
            prefer,
            req.get("holdings_ud_url"),
        )
        # summarize counts
        _add("holdings_from_urls", ok=True, counts={k: len(v) for k, v in accounts_before.items()})
    except Exception as e:
        _add("holdings_from_urls", ok=False, error=_exc_dict(e))
        return {"ok": False, "steps": steps}

    # 5) Load tag sheet (LEGENDS/ANA/DAL/LAK/PIT)
    try:
        tags = _load_player_tags(prefer, req.get("player_tags_url"))
        _add("load_player_tags", ok=True, tag_sizes={k: len(v) for k, v in tags.items()})
    except Exception as e:
        _add("load_player_tags", ok=False, error=_exc_dict(e))
        return {"ok": False, "steps": steps}

    # 6) Compute baseline family QP (sanity)
    try:
        fam_qp, per_qp, details = compute_family_qp(leader, accounts_before)
        _add("compute_family_qp", ok=True, family_qp_total=int(fam_qp))
    except Exception as e:
        _add("compute_family_qp", ok=False, error=_exc_dict(e))
        return {"ok": False, "steps": steps}

    return {"ok": True, "steps": steps}


# ---------- Small probe (confirms strict JSON is live) ----------

@app.get("/info")
def info():
    return {
        "ok": True,
        "title": getattr(app, "title", "Ultimate Quest Service (Small-Payload API)"),
        "version": getattr(app, "version", "unknown"),
        "default_response_class": "SafeJSONResponse",
    }
