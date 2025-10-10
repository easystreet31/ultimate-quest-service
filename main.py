# main.py
# Ultimate Quest Service — FastAPI app (v4.12.15)
#
# Full file — safe to copy/paste and replace your current main.py.
# - Registers ALL public routes used by the Quest Copilot (incl. Safe‑Sell + Delta Export).
# - Keeps the app thin: each route forwards the JSON/query to functions in app_core.
# - Uses graceful fallbacks: if a function name differs in app_core, we try a few aliases.
#
# Notes:
# - /family_safe_sell_report_by_urls was missing earlier; it’s wired here with name fallbacks.
# - /leaderboard_delta_export (CSV/XLSX) is included for large leaderboard diffs.
#
# Related docs & schemas you shared:
#   • User Guide macros & flows. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}
#   • Copilot Instructions (routes/output shape). :contentReference[oaicite:2]{index=2}
#   • Action JSON (OpenAPI) paths and request bodies. :contentReference[oaicite:3]{index=3}
#   • Render env defaults (links/knobs). :contentReference[oaicite:4]{index=4}
#   • Runtime pins (Procfile, Python version, requirements). :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}
#
from __future__ import annotations

import io
import os
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from fastapi import FastAPI, HTTPException, Request, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse, Response

# Import your core logic (do not fail hard if import-time side effects occur)
import app_core as _core  # type: ignore

__VERSION__ = "4.12.15"

app = FastAPI(
    title="Ultimate Quest Service (Small-Payload API)",
    version=__VERSION__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS (relaxed) ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Helpers ----------------------------------------------------------------------------------
def _first_callable(names: Sequence[str]) -> Callable[..., Any]:
    """
    Return the first callable attribute from app_core whose name matches one of `names`.
    Raise HTTP 501 if none is found.
    """
    for name in names:
        fn = getattr(_core, name, None)
        if callable(fn):
            return fn
    raise HTTPException(
        status_code=501,
        detail=f"Server function not found. Tried: {', '.join(names)}"
    )

def _defaults_from_env() -> Dict[str, Any]:
    """
    Minimal defaults fallback if app_core doesn't provide a defaults function.
    Shapes the same top-level keys you typically return.
    """
    def _get(key: str, default: Optional[str] = None) -> Optional[str]:
        v = os.getenv(key, default)
        return v.strip() if isinstance(v, str) else v

    rivals = _get("DEFAULT_TARGET_RIVALS", "") or ""
    rivals_list = [r.strip() for r in rivals.split(",") if r.strip()]

    return {
        "ok": True,
        "links": {
            "leaderboard": _get("DEFAULT_LEADERBOARD_URL"),
            "leaderboard_yday": _get("DEFAULT_LEADERBOARD_YDAY_URL"),
            "holdings_e31": _get("DEFAULT_HOLDINGS_E31_URL"),
            "holdings_dc": _get("DEFAULT_HOLDINGS_DC_URL"),
            "holdings_fe": _get("DEFAULT_HOLDINGS_FE_URL"),
            "holdings_ud": _get("DEFAULT_HOLDINGS_UD_URL"),
            "collection_e31": _get("DEFAULT_COLLECTION_E31_URL"),
            "collection_dc": _get("DEFAULT_COLLECTION_DC_URL"),
            "collection_fe": _get("DEFAULT_COLLECTION_FE_URL"),
            "collection_ud": _get("DEFAULT_COLLECTION_UD_URL"),
            "pool_collection": _get("DEFAULT_POOL_COLLECTION_URL"),
            "player_tags": _get("PLAYER_TAGS_URL"),
        },
        "rivals": rivals_list,
        "defend_buffer_default": int((_get("DEFAULT_DEFEND_BUFFER_ALL") or "15")),
        "force_family_urls": str(_get("FORCE_FAMILY_URLS", "true")).lower() == "true",
    }

def _safe_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Wrap core calls to surface readable HTTP errors instead of raw tracebacks.
    """
    try:
        return fn(*args, **kwargs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"type": type(e).__name__, "message": str(e)})

# --- Health/Info ------------------------------------------------------------------------------
@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "status": "healthy", "version": __VERSION__}

@app.get("/info")
def info() -> Dict[str, Any]:
    return {
        "ok": True,
        "title": "Ultimate Quest Service (Small-Payload API)",
        "version": __VERSION__,
        # We keep this string to match prior responses (no special class needed to use it)
        "default_response_class": "SafeJSONResponse",
    }

# --- Defaults ---------------------------------------------------------------------------------
@app.get("/defaults")
def get_defaults() -> Dict[str, Any]:
    # Prefer core's implementation if present; otherwise use env fallback.
    try:
        fn = _first_callable(["get_defaults", "defaults", "load_defaults"])
        return _safe_call(fn)
    except HTTPException as e:
        if e.status_code == 501:
            return _defaults_from_env()
        raise

# --- Probe (diagnostic) -----------------------------------------------------------------------
@app.post("/__probe_evaluate")
def __probe_evaluate(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    fn = _first_callable(["__probe_evaluate", "probe_evaluate", "probe_evaluate_by_urls"])
    return _safe_call(fn, payload)

# --- Evaluate trade ---------------------------------------------------------------------------
@app.post("/family_evaluate_trade_by_urls")
def family_evaluate_trade_by_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    fn = _first_callable(["family_evaluate_trade_by_urls", "evaluate_trade_by_urls", "family_evaluate_trade"])
    return _safe_call(fn, payload)

# --- Trade + Counter --------------------------------------------------------------------------
@app.post("/family_trade_plus_counter_by_urls")
def family_trade_plus_counter_by_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    fn = _first_callable(["family_trade_plus_counter_by_urls", "trade_plus_counter_by_urls", "family_trade_plus_counter"])
    return _safe_call(fn, payload)

# --- Collection Review (qty-aware) ------------------------------------------------------------
@app.post("/family_collection_review_by_urls")
def family_collection_review_by_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    fn = _first_callable(["family_collection_review_by_urls", "collection_review_by_urls", "family_collection_review"])
    return _safe_call(fn, payload)

# --- All-in (“own the whole sheet”) -----------------------------------------------------------
@app.post("/family_collection_all_in_by_urls")
def family_collection_all_in_by_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    fn = _first_callable(["family_collection_all_in_by_urls", "collection_all_in_by_urls", "family_collection_all_in"])
    return _safe_call(fn, payload)

# --- Safe‑Sell Report (NEW) -------------------------------------------------------------------
@app.post("/family_safe_sell_report_by_urls")
def family_safe_sell_report_by_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects (at minimum):
      {
        "prefer_env_defaults": true,
        "include_top3": false,
        "min_distance_to_rank3": 8,
        "exclude_accounts": []
      }
    Plus optional players_whitelist/players_blacklist and player_tags_url.
    """
    fn = _first_callable([
        "family_safe_sell_report_by_urls",
        "family_safe_sell_by_urls",
        "safe_sell_report_by_urls",
        "family_safe_sell_report",   # in case the core drops the suffix
    ])
    return _safe_call(fn, payload)

# --- Leaderboard Delta (JSON) -----------------------------------------------------------------
@app.post("/leaderboard_delta_by_urls")
def leaderboard_delta_by_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    fn = _first_callable(["leaderboard_delta_by_urls", "leaderboard_delta", "delta_by_urls"])
    return _safe_call(fn, payload)

# --- Leaderboard Delta Export (CSV/XLSX) ------------------------------------------------------
@app.get("/leaderboard_delta_export")
def leaderboard_delta_export(
    prefer_env_defaults: bool = Query(True),
    leaderboard_today_url: Optional[str] = Query(None),
    leaderboard_yesterday_url: Optional[str] = Query(None),
    rivals: Optional[str] = Query(None, description="Comma-separated usernames"),
    min_sp_delta: int = Query(1, ge=0),
    file_format: str = Query("csv", alias="format", pattern="^(csv|xlsx)$"),
) -> Response:
    """
    Export the leaderboard delta as CSV/XLSX. The core may return:
      • a Starlette Response/StreamingResponse,
      • a dict with {'content': bytes|str, 'mime': str, 'filename': str}, or
      • raw bytes/str (we'll wrap it).
    """
    params = {
        "prefer_env_defaults": prefer_env_defaults,
        "leaderboard_today_url": leaderboard_today_url,
        "leaderboard_yesterday_url": leaderboard_yesterday_url,
        "rivals": rivals,
        "min_sp_delta": min_sp_delta,
        "format": file_format,
    }
    fn = _first_callable([
        "leaderboard_delta_export",
        "export_leaderboard_delta",
        "leaderboard_delta_export_by_urls",
    ])
    res = _safe_call(fn, params)

    # If core already returns a Response (CSV/XLSX), pass it through.
    if isinstance(res, Response):
        return res

    # Dict contract: {'content': bytes|str, 'mime': str, 'filename': str}
    if isinstance(res, dict):
        content = res.get("content", b"")
        mime = res.get("mime") or ("text/csv" if file_format == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        filename = res.get("filename") or f"leaderboard_delta.{file_format}"
        if isinstance(content, str):
            content = content.encode("utf-8")
        return StreamingResponse(io.BytesIO(content), media_type=mime, headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        })

    # Raw bytes/str fallback
    if isinstance(res, (bytes, bytearray, str)):
        content_bytes = res if isinstance(res, (bytes, bytearray)) else str(res).encode("utf-8")
        mime = "text/csv" if file_format == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"leaderboard_delta.{file_format}"
        return StreamingResponse(io.BytesIO(content_bytes), media_type=mime, headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        })

    # Anything else: return JSON so the client can see what happened
    return JSONResponse(content={"ok": False, "detail": "Unexpected export return type", "preview": str(type(res))}, status_code=500)

# --- Root (optional) --------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "title": "Ultimate Quest Service (Small-Payload API)",
        "version": __VERSION__,
        "message": "See /docs for OpenAPI; use POST routes for evaluate/trade/counter/review/all-in/safe-sell; GET /leaderboard_delta_export for CSV/XLSX.",
    }

# --- Local dev entrypoint (Procfile uses uvicorn directly) ------------------------------------
if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
