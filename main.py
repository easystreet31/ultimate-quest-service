# main.py — bootstrap + safety middleware only
# Version: 4.1.0-reset

from __future__ import annotations
import os
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---- import the real app (your domain logic) ----
_boot_err = None
try:
    from app_core import app as core_app  # your real app with all routes
except Exception as e:
    _boot_err = e
    core_app = FastAPI(title="Ultimate Quest Service (boot fallback)")

app: FastAPI = core_app

# ---- env-backed defaults (read-only helper) ----
VERSION = os.getenv("UQS_VERSION", "4.1.0-reset")

DEFAULTS: Dict[str, Any] = {
    "leaderboard_url": os.getenv("DEFAULT_LEADERBOARD_URL"),
    "leaderboard_yesterday_url": os.getenv("DEFAULT_LEADERBOARD_YDAY_URL"),
    "holdings": {
        "E31": os.getenv("DEFAULT_HOLDINGS_E31_URL"),
        "DC": os.getenv("DEFAULT_HOLDINGS_DC_URL"),
        "FE": os.getenv("DEFAULT_HOLDINGS_FE_URL"),
    },
    "collections": {
        "E31": os.getenv("DEFAULT_COLLECTION_E31_URL"),
        "DC": os.getenv("DEFAULT_COLLECTION_DC_URL"),
        "FE": os.getenv("DEFAULT_COLLECTION_FE_URL"),
        "POOL": os.getenv("DEFAULT_POOL_COLLECTION_URL"),
    },
    "target_rivals": [x for x in os.getenv("DEFAULT_TARGET_RIVALS", "").split(",") if x],
    "defend_buffer_all": int(os.getenv("DEFAULT_DEFEND_BUFFER_ALL", "15") or "15"),
    "trade_fragility_default": os.getenv(
        "TRADE_FRAGILITY_DEFAULT", "trade_delta (strict; traded players only)"
    ),
    "force_family_urls": True,
}

def _has_route(path: str) -> bool:
    for r in app.routes:
        p = getattr(r, "path", None)
        if p == path:
            return True
    return False

# ---- /info (add only if app_core didn't define it) ----
if not _has_route("/info"):
    @app.get("/info")
    def info() -> Dict[str, Any]:
        paths = sorted({getattr(r, "path", None) for r in app.routes if getattr(r, "path", None)})
        return {"version": VERSION, "routes": paths}

# ---- /defaults (add only if app_core didn't define it) ----
if not _has_route("/defaults"):
    @app.get("/defaults")
    def defaults() -> Dict[str, Any]:
        return {**DEFAULTS, "version": VERSION}

# ---- safety middleware (trade fragility hardening + counter exclusions) ----
try:
    from baseline_gets_filter import BaselineGetsFilterMiddleware
    app.add_middleware(BaselineGetsFilterMiddleware)
except Exception as e:
    # If middleware import fails, expose a diagnostic route (won't break your app)
    if not _has_route("/__middleware_error"):
        @app.get("/__middleware_error")
        def middleware_error():
            return {"error": f"baseline_gets_filter import failed: {repr(e)}"}

# ---- if app_core import failed, expose a boot error (for Render logs) ----
if _boot_err and not _has_route("/__boot_error"):
    @app.get("/__boot_error")
    def boot_error():
        return {"error": f"Could not import app_core.app: {repr(_boot_err)}"}
