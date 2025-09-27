# main.py — wrapper around app_core.app
# Adds:
#   • Force-override of URLs with Render env defaults on all FAMILY endpoints (always).
#     - Also sanitizes placeholder tokens like "E31_default" anywhere.
#     - Optional escape hatch: set "allow_custom_urls": true in the request to disable overriding.
#   • Merge defaults on single-account endpoints; sanitize placeholders there too.
#   • Deep trade fragility filter: for trade endpoints, show only fragility created by the trade on traded players.
#   • /defaults diagnostic endpoint.

from __future__ import annotations
import os, json
from typing import Any, Dict, List, Tuple, Callable, Optional

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# ===== import the real app =====
_core_err = None
try:
    from app_core import app as core_app
except Exception as e:
    _core_err = e
    core_app = None

if core_app is None:
    raise RuntimeError(f"Could not import app_core.app: {_core_err!r}")

app: FastAPI = core_app

# ===== env helpers =====
def _env(k: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(k)
    return v if (v is not None and v != "") else default

def _to_int(v: Optional[str], default: int) -> int:
    try:
        return int(v) if v is not None else default
    except:
        return default

def _to_list_csv(v: Optional[str]) -> List[str]:
    if not v:
        return []
    return [p.strip() for p in v.split(",") if p.strip()]

# ===== URLs / defaults from env =====
DEFAULT_LEADERBOARD_URL       = _env("DEFAULT_LEADERBOARD_URL")
DEFAULT_LEADERBOARD_YDAY_URL  = _env("DEFAULT_LEADERBOARD_YDAY_URL")

DEFAULT_HOLDINGS_E31_URL      = _env("DEFAULT_HOLDINGS_E31_URL")
DEFAULT_HOLDINGS_DC_URL       = _env("DEFAULT_HOLDINGS_DC_URL")
DEFAULT_HOLDINGS_FE_URL       = _env("DEFAULT_HOLDINGS_FE_URL")

DEFAULT_COLLECTION_E31_URL    = _env("DEFAULT_COLLECTION_E31_URL")
DEFAULT_COLLECTION_DC_URL     = _env("DEFAULT_COLLECTION_DC_URL")
DEFAULT_COLLECTION_FE_URL     = _env("DEFAULT_COLLECTION_FE_URL")
DEFAULT_POOL_COLLECTION_URL   = _env("DEFAULT_POOL_COLLECTION_URL")

DEFAULT_TARGET_RIVALS         = _to_list_csv(_env("DEFAULT_TARGET_RIVALS", ""))
DEFAULT_DEFEND_BUFFER_ALL     = _to_int(_env("DEFAULT_DEFEND_BUFFER_ALL", "15"), 15)

# ===== endpoint groups =====
FAMILY_PATHS = {
    "/family_evaluate_trade_by_urls",
    "/collection_review_family_by_urls",
    "/family_transfer_suggestions_by_urls",
    "/family_transfer_optimize_by_urls",
}
SINGLE_PATHS = {
    "/evaluate_by_urls_easystreet31",
    "/scan_by_urls_easystreet31",
    "/scan_rival_by_urls_easystreet31",
    "/scan_partner_by_urls_easystreet31",
    "/review_collection_by_urls_easystreet31",
    "/suggest_give_from_collection_by_urls_easystreet31",
}
DELTA_PATHS = {"/leaderboard_delta_by_urls"}
TRADE_ENDPOINTS = {"/evaluate_by_urls_easystreet31", "/family_evaluate_trade_by_urls"}

# ===== placeholder sanitizing =====
PLACEHOLDER_TOKENS = {"E31_default", "DC_default", "FE_default", "LB_default", "POOL_default", "default_url"}

def _looks_bad_url(v: Any) -> bool:
    if not isinstance(v, str):
        return True
    s = v.strip()
    if not s:
        return True
    if any(tok.lower() in s.lower() for tok in PLACEHOLDER_TOKENS):
        return True
    # allow http(s) only
    if not s.startswith("http"):
        return True
    return False

def _sanitize_common_urls(body: Dict[str, Any]) -> None:
    # Single-account keys (if present)
    if _looks_bad_url(body.get("leaderboard_url")) and DEFAULT_LEADERBOARD_URL:
        body["leaderboard_url"] = DEFAULT_LEADERBOARD_URL
    if _looks_bad_url(body.get("holdings_url")) and DEFAULT_HOLDINGS_E31_URL:
        body["holdings_url"] = DEFAULT_HOLDINGS_E31_URL
    if _looks_bad_url(body.get("collection_url")) and DEFAULT_POOL_COLLECTION_URL:
        body["collection_url"] = DEFAULT_POOL_COLLECTION_URL
    if _looks_bad_url(body.get("my_collection_url")) and DEFAULT_COLLECTION_E31_URL:
        body["my_collection_url"] = DEFAULT_COLLECTION_E31_URL

def _force_family_urls(body: Dict[str, Any]) -> None:
    # Always stamp env defaults for family endpoints (unless allow_custom_urls is true)
    body["leaderboard_url"]    = DEFAULT_LEADERBOARD_URL
    body["holdings_e31_url"]   = DEFAULT_HOLDINGS_E31_URL
    body["holdings_dc_url"]    = DEFAULT_HOLDINGS_DC_URL
    body["holdings_fe_url"]    = DEFAULT_HOLDINGS_FE_URL
    body["collection_e31_url"] = DEFAULT_COLLECTION_E31_URL
    body["collection_dc_url"]  = DEFAULT_COLLECTION_DC_URL
    body["collection_fe_url"]  = DEFAULT_COLLECTION_FE_URL
    body["defend_buffer_all"]  = body.get("defend_buffer_all", DEFAULT_DEFEND_BUFFER_ALL)

def _merge_family_defaults(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_url", DEFAULT_LEADERBOARD_URL)
    body.setdefault("holdings_e31_url", DEFAULT_HOLDINGS_E31_URL)
    body.setdefault("holdings_dc_url", DEFAULT_HOLDINGS_DC_URL)
    body.setdefault("holdings_fe_url", DEFAULT_HOLDINGS_FE_URL)
    body.setdefault("collection_e31_url", DEFAULT_COLLECTION_E31_URL)
    body.setdefault("collection_dc_url", DEFAULT_COLLECTION_DC_URL)
    body.setdefault("collection_fe_url", DEFAULT_COLLECTION_FE_URL)
    body.setdefault("defend_buffer_all", DEFAULT_DEFEND_BUFFER_ALL)

def _merge_single_defaults(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_url", DEFAULT_LEADERBOARD_URL)
    body.setdefault("holdings_url", DEFAULT_HOLDINGS_E31_URL)
    body.setdefault("collection_url", DEFAULT_POOL_COLLECTION_URL)
    body.setdefault("my_collection_url", DEFAULT_COLLECTION_E31_URL)
    body.setdefault("defend_buffer", DEFAULT_DEFEND_BUFFER_ALL)
    if "target_rivals" not in body and DEFAULT_TARGET_RIVALS:
        body["target_rivals"] = DEFAULT_TARGET_RIVALS

def _merge_delta_defaults(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_today_url",     DEFAULT_LEADERBOARD_URL)
    body.setdefault("leaderboard_yesterday_url", DEFAULT_LEADERBOARD_YDAY_URL)
    body.setdefault("leaderboard_url", body.get("leaderboard_today_url") or DEFAULT_LEADERBOARD_URL)

# ===== middleware: merge/override + fragility filter =====
class MergeDefaultsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method.upper() != "POST":
            return await call_next(request)

        path = request.url.path
        try:
            raw = await request.body()
            body = {} if not raw else json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                return await call_next(request)
        except Exception:
            return await call_next(request)

        # Base merges
        if path in FAMILY_PATHS:
            # Always override with env defaults, unless the caller opts out explicitly.
            allow_custom = bool(body.get("allow_custom_urls") is True)
            if allow_custom:
                _merge_family_defaults(body)
            else:
                _force_family_urls(body)
        elif path in SINGLE_PATHS:
            _merge_single_defaults(body)
            _sanitize_common_urls(body)
        elif path in DELTA_PATHS:
            _merge_delta_defaults(body)

        # Default fragility_mode for trade calls (if not specified)
        if path in TRADE_ENDPOINTS and body.get("fragility_mode") is None:
            body["fragility_mode"] = "trade_delta"

        # Save merged body so downstream can read it (fragility filter too)
        request.state.merged_body = body
        new_raw = json.dumps(body).encode("utf-8")

        async def receive():
            return {"type": "http.request", "body": new_raw, "more_body": False}
        request._receive = receive  # type: ignore[attr-defined]

        return await call_next(request)

app.add_middleware(MergeDefaultsMiddleware)

# ===== Deep Trade Fragility Filter =====
ALLOWED_FRAG_MODES = {"none", "trade_only", "trade_delta"}  # "all" is not honored on trade endpoints

def _expand_trade_players(body: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for line in body.get("trade") or []:
        players = line.get("players")
        if not isinstance(players, str):
            continue
        parts = [s.strip() for s in players.replace(",", "/").split("/") if s.strip()]
        out.extend(parts if parts else [players.strip()])
    # unique, case-insensitive
    seen, uniq = set(), []
    for p in out:
        k = p.lower()
        if k not in seen:
            seen.add(k); uniq.append(p)
    return uniq

def _read_margin(d: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    b = d.get("margin_before", d.get("before_margin"))
    a = d.get("margin_after",  d.get("after_margin", d.get("margin")))
    try: b = float(b) if b is not None else None
    except: b = None
    try: a = float(a) if a is not None else None
    except: a = None
    return b, a

def _get_name(x: Dict[str, Any]) -> Optional[str]:
    nm = x.get("player") or x.get("name") or x.get("players")
    return nm if isinstance(nm, str) and nm.strip() else None

def _looks_frag_list(lst: Any) -> bool:
    if not isinstance(lst, list) or not lst:
        return False
    dicts = [x for x in lst if isinstance(x, dict)]
    if not dicts:
        return False
    if not any(_get_name(x) for x in dicts):
        return False
    for x in dicts:
        if any(k in x for k in ("margin","margin_before","margin_after","before_margin","after_margin","fragile","is_fragile","status")):
            return True
    return False

def _keep_item(x: Dict[str, Any], tset: set, buffer_: int, mode: str) -> bool:
    nm = _get_name(x)
    if not nm or nm.lower() not in tset:
        return False
    b, a = _read_margin(x)
    if a is None and b is None:
        return False
    if mode == "trade_delta":
        if a is None or b is None:
            return bool(a is not None and a <= buffer_)  # degrade to trade_only
        return (b > buffer_) and (a <= buffer_)
    if mode == "trade_only":
        return bool(a is not None and a <= buffer_)
    return False  # "none" or anything else

def _deep_filter(obj: Any, tset: set, buffer_: int, mode: str) -> Any:
    if isinstance(obj, list):
        if _looks_frag_list(obj):
            return [x for x in obj if isinstance(x, dict) and _keep_item(x, tset, buffer_, mode)]
        return [_deep_filter(x, tset, buffer_, mode) for x in obj]
    if isinstance(obj, dict):
        nd: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, list) and _looks_frag_list(v):
                nd[k] = [x for x in v if isinstance(x, dict) and _keep_item(x, tset, buffer_, mode)]
            else:
                nd[k] = _deep_filter(v, tset, buffer_, mode)
        return nd
    return obj

class FragilityFilterMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        try:
            path = request.url.path
            if request.method.upper() != "POST":  # only POSTs have bodies/results for our endpoints
                return response
            if path not in TRADE_ENDPOINTS:
                return response
            if not getattr(request, "state", None) or not isinstance(request.state.merged_body, dict):
                return response

            body = request.state.merged_body
            mode = str(body.get("fragility_mode") or "trade_delta").lower()
            if mode not in ALLOWED_FRAG_MODES:
                mode = "trade_delta"
            defend_buffer = int(body.get("defend_buffer") or body.get("defend_buffer_all") or DEFAULT_DEFEND_BUFFER_ALL or 15)
            tset = {p.lower() for p in _expand_trade_players(body)}
            if not tset:
                return response

            # read original response body (consume iterator if present)
            raw = b""
            if hasattr(response, "body_iterator"):
                chunks = []
                async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                    chunks.append(chunk)
                raw = b"".join(chunks)
            else:
                raw = getattr(response, "body", b"") or b""

            if not raw:
                return response

            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                return Response(content=raw, status_code=response.status_code, media_type=response.media_type)

            if not isinstance(payload, dict):
                return Response(content=raw, status_code=response.status_code, media_type="application/json")

            filtered = _deep_filter(payload, tset, defend_buffer, mode)
            notes = filtered.setdefault("_notes", [])
            if isinstance(notes, list):
                notes.append({
                    "fragility_filter": mode,
                    "trade_players": sorted(list(tset)),
                    "defend_buffer_used": defend_buffer
                })

            new_raw = json.dumps(filtered, ensure_ascii=False).encode("utf-8")
            return Response(content=new_raw, status_code=response.status_code, media_type="application/json")
        except Exception:
            return response

app.add_middleware(FragilityFilterMiddleware)

# ===== diagnostics =====
@app.get("/defaults")
def defaults():
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
        "trade_fragility_default": "trade_delta (deep, traded players only)",
        "force_family_urls": True,
    }
