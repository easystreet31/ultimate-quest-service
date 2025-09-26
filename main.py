"""
main.py — thin wrapper that injects default URLs & knobs from environment variables
into request bodies when callers omit them, then forwards to your original app.

HOW TO USE:
1) Rename your existing working app file from `main.py` to `app_core.py`.
2) Place this file as `main.py`.
3) No other changes needed. Start command remains: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Callable

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

# ------------------------------------------------------------------------------
# 1) Import your original app (now in app_core.py)
# ------------------------------------------------------------------------------
try:
    import app_core  # <-- this is your previous `main.py` renamed to `app_core.py`
except Exception as exc:
    # Give a very clear error if the rename step was missed.
    raise RuntimeError(
        "Could not import `app_core`. Please rename your previous working main.py to app_core.py"
    ) from exc

app: FastAPI = app_core.app  # use your existing FastAPI instance


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
# 3) Known POST endpoints that can accept defaults when a field is omitted
# ------------------------------------------------------------------------------

# NOTE: We *do not* change bodies that already include a field; we only fill in missing ones.
# This keeps behavior 100% backward compatible with your working logic.

POST_PATHS: Dict[str, Callable[[Dict[str, Any]], None]] = {}

def _merge_list(value: Any) -> Optional[List[str]]:
    """Coerce CSV string → list[str], pass through list, ignore otherwise."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    return None

def _apply_common_defaults(body: Dict[str, Any]) -> None:
    # single-account convenience
    body.setdefault("leaderboard_url", DEFAULT_LEADERBOARD_URL)
    body.setdefault("holdings_url", DEFAULT_HOLDINGS_E31_URL)

    # partner/seller collections
    body.setdefault("collection_url", DEFAULT_POOL_COLLECTION_URL)
    body.setdefault("my_collection_url", DEFAULT_COLLECTION_E31_URL)

    # rival targets
    if not body.get("target_rivals"):
        if DEFAULT_TARGET_RIVALS:
            body["target_rivals"] = DEFAULT_TARGET_RIVALS
    else:
        # normalize if user passed CSV string
        maybe = _merge_list(body.get("target_rivals"))
        if maybe is not None:
            body["target_rivals"] = maybe

    # default defend buffer knobs (only where a single value is expected)
    body.setdefault("defend_buffer", DEFAULT_DEFEND_BUFFER_ALL)

def _apply_family_defaults(body: Dict[str, Any]) -> None:
    # family (three accounts)
    body.setdefault("leaderboard_url", DEFAULT_LEADERBOARD_URL)

    body.setdefault("holdings_e31_url", DEFAULT_HOLDINGS_E31_URL)
    body.setdefault("holdings_dc_url",  DEFAULT_HOLDINGS_DC_URL)
    body.setdefault("holdings_fe_url",  DEFAULT_HOLDINGS_FE_URL)

    body.setdefault("collection_e31_url", DEFAULT_COLLECTION_E31_URL)
    body.setdefault("collection_dc_url",  DEFAULT_COLLECTION_DC_URL)
    body.setdefault("collection_fe_url",  DEFAULT_COLLECTION_FE_URL)

    # default defend buffer for "all" if not provided
    body.setdefault("defend_buffer_all", DEFAULT_DEFEND_BUFFER_ALL)

# Endpoint-specific mergers
def _merge_evaluate(body: Dict[str, Any]) -> None:
    _apply_common_defaults(body)

def _merge_scan(body: Dict[str, Any]) -> None:
    _apply_common_defaults(body)

def _merge_scan_rival(body: Dict[str, Any]) -> None:
    _apply_common_defaults(body)
    # Normalize focus_rival to a simple string if someone passes ["name"]
    if isinstance(body.get("focus_rival"), list):
        arr = [s for s in body["focus_rival"] if isinstance(s, str) and s.strip()]
        if arr:
            body["focus_rival"] = arr[0]

def _merge_partner(body: Dict[str, Any]) -> None:
    _apply_common_defaults(body)

def _merge_review_collection(body: Dict[str, Any]) -> None:
    _apply_common_defaults(body)

def _merge_safe_give(body: Dict[str, Any]) -> None:
    _apply_common_defaults(body)

def _merge_family_eval_trade(body: Dict[str, Any]) -> None:
    _apply_family_defaults(body)

def _merge_collection_family(body: Dict[str, Any]) -> None:
    _apply_family_defaults(body)

def _merge_leaderboard_delta(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_today_url",     DEFAULT_LEADERBOARD_URL)
    body.setdefault("leaderboard_yesterday_url", DEFAULT_LEADERBOARD_YDAY_URL)
    # also allow body.leaderboard_url as alias for "today"
    if not body.get("leaderboard_today_url"):
        body["leaderboard_today_url"] = body.get("leaderboard_url") or DEFAULT_LEADERBOARD_URL

def _merge_family_transfers(body: Dict[str, Any]) -> None:
    _apply_family_defaults(body)

# Register the endpoints you use (paths must match your server exactly)
POST_PATHS["/evaluate_by_urls_easystreet31"]          = _merge_evaluate
POST_PATHS["/scan_by_urls_easystreet31"]              = _merge_scan
POST_PATHS["/scan_rival_by_urls_easystreet31"]        = _merge_scan_rival
POST_PATHS["/scan_partner_by_urls_easystreet31"]      = _merge_partner
POST_PATHS["/review_collection_by_urls_easystreet31"] = _merge_review_collection
POST_PATHS["/suggest_give_from_collection_by_urls_easystreet31"] = _merge_safe_give

# Family routes
POST_PATHS["/family_evaluate_trade_by_urls"]          = _merge_family_eval_trade
POST_PATHS["/collection_review_family_by_urls"]       = _merge_collection_family
POST_PATHS["/family_transfer_suggestions_by_urls"]    = _merge_family_transfers
POST_PATHS["/family_transfer_optimize_by_urls"]       = _merge_family_transfers

# Deltas
POST_PATHS["/leaderboard_delta_by_urls"]              = _merge_leaderboard_delta


# ------------------------------------------------------------------------------
# 4) Middleware to inject defaults when a client omitted them
# ------------------------------------------------------------------------------

class MergeDefaultsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Only for POST JSON requests targeting one of the known endpoints.
        if request.method.upper() != "POST":
            return await call_next(request)

        path = request.url.path
        merger = POST_PATHS.get(path)
        if not merger:
            return await call_next(request)

        # Read body once
        try:
            raw = await request.body()
            if not raw:
                body = {}
            else:
                body = json.loads(raw.decode("utf-8"))
                if not isinstance(body, dict):
                    # If a different schema (e.g., list) comes in, don't modify.
                    return await call_next(request)
        except Exception:
            return await call_next(request)

        # Apply defaults
        try:
            merger(body)
        except Exception:
            # Never block a request due to default merge logic—just pass through.
            return await call_next(request)

        # Re-inject modified body
        new_raw = json.dumps(body).encode("utf-8")

        async def receive() -> dict:
            return {"type": "http.request", "body": new_raw, "more_body": False}

        # Monkey-patch the request stream for downstream handlers
        request._receive = receive  # type: ignore[attr-defined]

        return await call_next(request)

# Attach the middleware to YOUR existing app
app.add_middleware(MergeDefaultsMiddleware)


# ------------------------------------------------------------------------------
# 5) Optional: simple /defaults echo endpoint for quick checks (non-invasive)
# ------------------------------------------------------------------------------

@app.get("/defaults")
def defaults_echo() -> Dict[str, Any]:
    """Quick diagnostic: returns which defaults are currently active."""
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
        "note": "Defaults only apply when a field is omitted in the request body."
    }
