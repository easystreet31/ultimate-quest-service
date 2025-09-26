"""
main.py — wrapper that injects default URLs/knobs from environment variables
into request bodies (only when fields are omitted), then forwards to your
original FastAPI app.

HOW TO DEPLOY:
1) Rename your previous working file from `main.py` to `app_core.py` (same directory).
2) Add this file as `main.py`.
3) No other changes needed. Start command remains: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Callable

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

# ------------------------------------------------------------------------------
# 1) Import your original app (now in app_core.py)
# ------------------------------------------------------------------------------
_core_import_error = None
core_app = None
try:
    # Your previous app file, renamed.
    from app_core import app as core_app  # type: ignore
except Exception as e:
    _core_import_error = e

if core_app is None:
    # Helpful guidance if the rename was missed or path is wrong.
    raise RuntimeError(
        "Could not import your original FastAPI app from `app_core.py`.\n"
        "Please rename your previous working `main.py` to `app_core.py` and commit it at the repo root.\n"
        f"Original import error: {repr(_core_import_error)}"
    )

# This is the FastAPI instance uvicorn will serve.
app: FastAPI = core_app


# ------------------------------------------------------------------------------
# 2) Read environment defaults once on startup
# ------------------------------------------------------------------------------

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if (v is not None and v != "") else default

DEFAULT_LEADERBOARD_URL      = _env("DEFAULT_LEADERBOARD_URL")
DEFAULT_LEADERBOARD_YDAY_URL = _env("DEFAULT_LEADERBOARD_YDAY_URL")

DEFAULT_HOLDINGS_E31_URL     = _env("DEFAULT_HOLDINGS_E31_URL")
DEFAULT_HOLDINGS_DC_URL      = _env("DEFAULT_HOLDINGS_DC_URL")
DEFAULT_HOLDINGS_FE_URL      = _env("DEFAULT_HOLDINGS_FE_URL")

DEFAULT_COLLECTION_E31_URL   = _env("DEFAULT_COLLECTION_E31_URL")
DEFAULT_COLLECTION_DC_URL    = _env("DEFAULT_COLLECTION_DC_URL")
DEFAULT_COLLECTION_FE_URL    = _env("DEFAULT_COLLECTION_FE_URL")
DEFAULT_POOL_COLLECTION_URL  = _env("DEFAULT_POOL_COLLECTION_URL")

DEFAULT_TARGET_RIVALS_RAW    = _env("DEFAULT_TARGET_RIVALS", "")
DEFAULT_TARGET_RIVALS: List[str] = (
    [r.strip() for r in DEFAULT_TARGET_RIVALS_RAW.split(",") if r.strip()]
    if DEFAULT_TARGET_RIVALS_RAW is not None
    else []
)

def _to_int(val: Optional[str], fallback: int) -> int:
    try:
        return int(val) if val is not None else fallback
    except Exception:
        return fallback

DEFAULT_DEFEND_BUFFER_ALL = _to_int(_env("DEFAULT_DEFEND_BUFFER_ALL", "15"), 15)


# ------------------------------------------------------------------------------
# 3) Endpoint-specific default mergers (called only if a field is missing)
# ------------------------------------------------------------------------------

def _merge_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    return None

def _apply_common_defaults(body: Dict[str, Any]) -> None:
    # Single-account convenience
    body.setdefault("leaderboard_url", DEFAULT_LEADERBOARD_URL)
    body.setdefault("holdings_url",    DEFAULT_HOLDINGS_E31_URL)

    # Partner/seller collections
    body.setdefault("collection_url",    DEFAULT_POOL_COLLECTION_URL)
    body.setdefault("my_collection_url", DEFAULT_COLLECTION_E31_URL)

    # Rivals
    if not body.get("target_rivals"):
        if DEFAULT_TARGET_RIVALS:
            body["target_rivals"] = DEFAULT_TARGET_RIVALS
    else:
        maybe = _merge_list(body.get("target_rivals"))
        if maybe is not None:
            body["target_rivals"] = maybe

    # Defend buffer
    body.setdefault("defend_buffer", DEFAULT_DEFEND_BUFFER_ALL)

def _apply_family_defaults(body: Dict[str, Any]) -> None:
    # Three-account defaults
    body.setdefault("leaderboard_url", DEFAULT_LEADERBOARD_URL)

    body.setdefault("holdings_e31_url", DEFAULT_HOLDINGS_E31_URL)
    body.setdefault("holdings_dc_url",  DEFAULT_HOLDINGS_DC_URL)
    body.setdefault("holdings_fe_url",  DEFAULT_HOLDINGS_FE_URL)

    body.setdefault("collection_e31_url", DEFAULT_COLLECTION_E31_URL)
    body.setdefault("collection_dc_url",  DEFAULT_COLLECTION_DC_URL)
    body.setdefault("collection_fe_url",  DEFAULT_COLLECTION_FE_URL)

    body.setdefault("defend_buffer_all", DEFAULT_DEFEND_BUFFER_ALL)

def _merge_evaluate(body: Dict[str, Any]) -> None:               _apply_common_defaults(body)
def _merge_scan(body: Dict[str, Any]) -> None:                   _apply_common_defaults(body)
def _merge_scan_rival(body: Dict[str, Any]) -> None:
    _apply_common_defaults(body)
    if isinstance(body.get("focus_rival"), list):
        arr = [s for s in body["focus_rival"] if isinstance(s, str) and s.strip()]
        if arr:
            body["focus_rival"] = arr[0]
def _merge_partner(body: Dict[str, Any]) -> None:                _apply_common_defaults(body)
def _merge_review_collection(body: Dict[str, Any]) -> None:      _apply_common_defaults(body)
def _merge_safe_give(body: Dict[str, Any]) -> None:              _apply_common_defaults(body)
def _merge_family_eval_trade(body: Dict[str, Any]) -> None:      _apply_family_defaults(body)
def _merge_collection_family(body: Dict[str, Any]) -> None:      _apply_family_defaults(body)
def _merge_family_transfers(body: Dict[str, Any]) -> None:       _apply_family_defaults(body)
def _merge_leaderboard_delta(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_today_url",     DEFAULT_LEADERBOARD_URL)
    body.setdefault("leaderboard_yesterday_url", DEFAULT_LEADERBOARD_YDAY_URL)
    if not body.get("leaderboard_today_url"):
        body["leaderboard_today_url"] = body.get("leaderboard_url") or DEFAULT_LEADERBOARD_URL

# Map exactly to your server routes
POST_PATHS: Dict[str, Callable[[Dict[str, Any]], None]] = {
    "/evaluate_by_urls_easystreet31":          _merge_evaluate,
    "/scan_by_urls_easystreet31":              _merge_scan,
    "/scan_rival_by_urls_easystreet31":        _merge_scan_rival,
    "/scan_partner_by_urls_easystreet31":      _merge_partner,
    "/review_collection_by_urls_easystreet31": _merge_review_collection,
    "/suggest_give_from_collection_by_urls_easystreet31": _merge_safe_give,
    "/family_evaluate_trade_by_urls":          _merge_family_eval_trade,
    "/collection_review_family_by_urls":       _merge_collection_family,
    "/family_transfer_suggestions_by_urls":    _merge_family_transfers,
    "/family_transfer_optimize_by_urls":       _merge_family_transfers,
    "/leaderboard_delta_by_urls":              _merge_leaderboard_delta,
}


# ------------------------------------------------------------------------------
# 4) Middleware to inject defaults when a client omitted them
# ------------------------------------------------------------------------------

class MergeDefaultsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method.upper() != "POST":
            return await call_next(request)

        path = request.url.path
        merger = POST_PATHS.get(path)
        if not merger:
            return await call_next(request)

        try:
            raw = await request.body()
            body: Dict[str, Any]
            if not raw:
                body = {}
            else:
                body_json = json.loads(raw.decode("utf-8"))
                if not isinstance(body_json, dict):
                    # Only merge into object bodies; otherwise pass through unchanged.
                    return await call_next(request)
                body = body_json
        except Exception:
            return await call_next(request)

        # Apply defaults (no override of user-provided fields)
        try:
            merger(body)
        except Exception:
            # Never block a request due to default-merger issues—pass through.
            return await call_next(request)

        # Re-inject the modified body for downstream handlers
        new_raw = json.dumps(body).encode("utf-8")

        async def receive():
            return {"type": "http.request", "body": new_raw, "more_body": False}

        # Monkey-patch the request receive stream
        request._receive = receive  # type: ignore[attr-defined]

        return await call_next(request)

# Attach middleware to your existing app
app.add_middleware(MergeDefaultsMiddleware)


# ------------------------------------------------------------------------------
# 5) Diagnostics — quick way to confirm defaults are active
# ------------------------------------------------------------------------------

@app.get("/defaults")
def defaults_echo() -> Dict[str, Any]:
    return {
        "leaderboard_url": DEFAULT_LEADERBOARD_URL,
        "leaderboard_yesterday_url": DEFAULT_LEADERBOARD_YDAY_URL,
        "holdings": {
            "E31": DEFAULT_HOLDINGS_E31_URL,
            "DC":  DEFAULT_HOLDINGS_DC_URL,
            "FE":  DEFAULT_HOLDINGS_FE_URL,
        },
        "collections": {
            "E31": DEFAULT_COLLECTION_E31_URL,
            "DC":  DEFAULT_COLLECTION_DC_URL,
            "FE":  DEFAULT_COLLECTION_FE_URL,
            "POOL": DEFAULT_POOL_COLLECTION_URL,
        },
        "target_rivals": DEFAULT_TARGET_RIVALS,
        "defend_buffer_all": DEFAULT_DEFEND_BUFFER_ALL,
        "note": "Defaults only apply when a field is omitted in the POST body."
    }
