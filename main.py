# main.py
# Ultimate Quest Service — FastAPI app (v4.12.16)
#
# Full file — copy/paste to replace your current main.py.
# - Registers ALL public routes (incl. Safe‑Sell + Delta Export).
# - Thin controller: forwards to functions in app_core.
# - Safe‑Sell now has a built‑in FALLBACK implementation if app_core lacks it.
#
from __future__ import annotations

import io
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse, Response

# Import your core logic
import app_core as _core  # type: ignore

__VERSION__ = "4.12.16"

app = FastAPI(
    title="Ultimate Quest Service (Small-Payload API)",
    version=__VERSION__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS -------------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Helpers ----------------------------------------------------------------------------------
def _first_callable(names: Sequence[str]) -> Callable[..., Any]:
    """Return the first callable attribute from app_core whose name matches one of `names`.
    Raise HTTP 501 if none is found."""
    for name in names:
        fn = getattr(_core, name, None)
        if callable(fn):
            return fn
    raise HTTPException(
        status_code=501,
        detail=f"Server function not found. Tried: {', '.join(names)}"
    )

def _defaults_from_env() -> Dict[str, Any]:
    """Minimal defaults fallback if app_core doesn't provide a defaults function."""
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
            "holdings_dc":  _get("DEFAULT_HOLDINGS_DC_URL"),
            "holdings_fe":  _get("DEFAULT_HOLDINGS_FE_URL"),
            "holdings_ud":  _get("DEFAULT_HOLDINGS_UD_URL"),
            "collection_e31": _get("DEFAULT_COLLECTION_E31_URL"),
            "collection_dc":  _get("DEFAULT_COLLECTION_DC_URL"),
            "collection_fe":  _get("DEFAULT_COLLECTION_FE_URL"),
            "collection_ud":  _get("DEFAULT_COLLECTION_UD_URL"),
            "pool_collection": _get("DEFAULT_POOL_COLLECTION_URL"),
            "player_tags": _get("PLAYER_TAGS_URL"),
        },
        "rivals": rivals_list,
        "defend_buffer_default": int((_get("DEFAULT_DEFEND_BUFFER_ALL") or "15")),
        "force_family_urls": str(_get("FORCE_FAMILY_URLS", "true")).lower() == "true",
    }

def _safe_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Wrap core calls to surface readable HTTP errors."""
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
        "default_response_class": "SafeJSONResponse",
    }

# --- Defaults ---------------------------------------------------------------------------------
@app.get("/defaults")
def get_defaults() -> Dict[str, Any]:
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

# --- Safe‑Sell Report (with fallback) ---------------------------------------------------------
@app.post("/family_safe_sell_report_by_urls")
def family_safe_sell_report_by_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects at minimum:
      {
        "prefer_env_defaults": true,
        "include_top3": false,
        "min_distance_to_rank3": 8,
        "exclude_accounts": []
      }
    Optional: players_whitelist, players_blacklist, player_tags_url, leaderboard_url,
              holdings_*_url (when not using defaults).
    """
    # Try app_core first
    try:
        fn = _first_callable([
            "family_safe_sell_report_by_urls",
            "family_safe_sell_by_urls",
            "safe_sell_report_by_urls",
            "family_safe_sell_report",
        ])
        return _safe_call(fn, payload)
    except HTTPException as e:
        if e.status_code != 501:
            raise  # real error from core; bubble up

    # ---------- Fallback implementation (uses app_core helpers) ----------
    prefer_env_defaults = bool(payload.get("prefer_env_defaults", True))
    include_top3 = bool(payload.get("include_top3", False))
    min_dist = int(payload.get("min_distance_to_rank3", 6))
    exclude_accounts: List[str] = payload.get("exclude_accounts") or []
    wl = {str(p).lower() for p in (payload.get("players_whitelist") or [])}
    bl = {str(p).lower() for p in (payload.get("players_blacklist") or [])}

    # Load leaderboard (today) and holdings via core utilities
    lb_url = _core._pick_url(payload.get("leaderboard_url"), "leaderboard", prefer_env_defaults)
    leader = _core.normalize_leaderboard(_core.fetch_xlsx(lb_url))

    accounts = _core.holdings_from_urls(
        payload.get("holdings_e31_url"),
        payload.get("holdings_dc_url"),
        payload.get("holdings_fe_url"),
        prefer_env_defaults,
        payload.get("holdings_ud_url"),
    )

    items: List[Dict[str, Any]] = []
    FAMILY_ACCOUNTS = getattr(_core, "FAMILY_ACCOUNTS", ["Easystreet31", "DusterCrusher", "FinkleIsEinhorn", "UpperDuck"])

    for acct in FAMILY_ACCOUNTS:
        if acct in exclude_accounts:
            continue
        for player, sp_owned in sorted(accounts.get(acct, {}).items()):
            if sp_owned <= 0:
                continue
            p_lower = (player or "").lower()
            if wl and p_lower not in wl:
                continue
            if bl and p_lower in bl:
                continue

            rows = _core._smallset_entries_for_player(player, leader, accounts)
            if not rows:
                continue
            ordered, rank_by_key = _core._dedup_and_rank(rows)
            r, s = rank_by_key.get(_core._canon_user_strong(acct), (9999, 0))
            if (not include_top3) and r <= 3:
                continue

            # Determine the SP threshold for Rank‑3
            if len(ordered) > 2:
                third_sp = ordered[2][2]
            elif len(ordered) > 1:
                third_sp = ordered[1][2]
            else:
                third_sp = 0
            distance_to_rank3 = max((third_sp + 1) - s, 0)

            if distance_to_rank3 < min_dist:
                continue

            items.append({
                "account": acct,
                "player": player,
                "sp_owned": int(sp_owned),
                "best_rank": int(r),
                "distance_to_rank3": int(distance_to_rank3),
            })

    # Sort: farthest from Rank‑3 first (safest to sell), then alpha
    items.sort(key=lambda x: (-x["distance_to_rank3"], x["account"], x["player"].lower()))

    return {"ok": True, "count": len(items), "items": items, "method": "fallback_main.py"}

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

# --- Root -------------------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "title": "Ultimate Quest Service (Small-Payload API)",
        "version": __VERSION__,
        "message": "See /docs for OpenAPI; use POST routes for evaluate/trade/counter/review/all-in/safe-sell; GET /leaderboard_delta_export for CSV/XLSX.",
    }

# --- Local dev entrypoint ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
