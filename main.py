# main.py — wrapper around your core FastAPI app (app_core.py)
# Features:
#   • MergeDefaultsMiddleware — fills/overrides request with env defaults (supports prefer_env_defaults=true)
#   • FragilityFilterMiddleware — for trade endpoints, show only trade-created fragility on traded players
#       - deep, recursive filtering (handles nested lists)
#       - coerces unexpected modes (e.g., "all") to "trade_delta" unless explicitly "none"/"trade_only"/"trade_delta"
#   • /defaults endpoint for quick diagnostics of current env URLs/settings

from __future__ import annotations
import os, json
from typing import Any, Dict, List, Tuple, Callable, Optional

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# ========= import the real app =========
_core_err = None
try:
    from app_core import app as core_app
except Exception as e:
    _core_err = e
    core_app = None

if core_app is None:
    raise RuntimeError(f"Could not import app_core.app: {_core_err!r}")

app: FastAPI = core_app

# ========= env helpers =========
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

# ========= URLs / defaults from env =========
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

# ========= request merging / override =========
def _merge_common(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_url", DEFAULT_LEADERBOARD_URL)
    body.setdefault("holdings_url", DEFAULT_HOLDINGS_E31_URL)
    body.setdefault("collection_url", DEFAULT_POOL_COLLECTION_URL)
    body.setdefault("my_collection_url", DEFAULT_COLLECTION_E31_URL)
    body.setdefault("defend_buffer", DEFAULT_DEFEND_BUFFER_ALL)
    if "target_rivals" not in body and DEFAULT_TARGET_RIVALS:
        body["target_rivals"] = DEFAULT_TARGET_RIVALS
    # default fragility behavior for trade eval (only when caller didn’t specify)
    if body.get("fragility_mode") is None and body.get("trade"):
        body["fragility_mode"] = "trade_delta"

def _override_common(body: Dict[str, Any]) -> None:
    body["leaderboard_url"] = DEFAULT_LEADERBOARD_URL
    body["holdings_url"] = DEFAULT_HOLDINGS_E31_URL
    body["collection_url"] = DEFAULT_POOL_COLLECTION_URL
    body["my_collection_url"] = DEFAULT_COLLECTION_E31_URL
    body["defend_buffer"] = DEFAULT_DEFEND_BUFFER_ALL
    if DEFAULT_TARGET_RIVALS:
        body["target_rivals"] = DEFAULT_TARGET_RIVALS
    if body.get("trade") and body.get("fragility_mode") is None:
        body["fragility_mode"] = "trade_delta"

def _merge_family(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_url", DEFAULT_LEADERBOARD_URL)
    body.setdefault("holdings_e31_url", DEFAULT_HOLDINGS_E31_URL)
    body.setdefault("holdings_dc_url",  DEFAULT_HOLDINGS_DC_URL)
    body.setdefault("holdings_fe_url",  DEFAULT_HOLDINGS_FE_URL)
    body.setdefault("collection_e31_url", DEFAULT_COLLECTION_E31_URL)
    body.setdefault("collection_dc_url",  DEFAULT_COLLECTION_DC_URL)
    body.setdefault("collection_fe_url",  DEFAULT_COLLECTION_FE_URL)
    body.setdefault("defend_buffer_all", DEFAULT_DEFEND_BUFFER_ALL)
    if body.get("fragility_mode") is None and body.get("trade"):
        body["fragility_mode"] = "trade_delta"

def _override_family(body: Dict[str, Any]) -> None:
    body["leaderboard_url"]    = DEFAULT_LEADERBOARD_URL
    body["holdings_e31_url"]   = DEFAULT_HOLDINGS_E31_URL
    body["holdings_dc_url"]    = DEFAULT_HOLDINGS_DC_URL
    body["holdings_fe_url"]    = DEFAULT_HOLDINGS_FE_URL
    body["collection_e31_url"] = DEFAULT_COLLECTION_E31_URL
    body["collection_dc_url"]  = DEFAULT_COLLECTION_DC_URL
    body["collection_fe_url"]  = DEFAULT_COLLECTION_FE_URL
    body["defend_buffer_all"]  = DEFAULT_DEFEND_BUFFER_ALL
    if body.get("trade") and body.get("fragility_mode") is None:
        body["fragility_mode"] = "trade_delta"

def _merge_delta(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_today_url",     DEFAULT_LEADERBOARD_URL)
    body.setdefault("leaderboard_yesterday_url", DEFAULT_LEADERBOARD_YDAY_URL)
    body.setdefault("leaderboard_url",           body.get("leaderboard_today_url") or DEFAULT_LEADERBOARD_URL)

def _override_delta(body: Dict[str, Any]) -> None:
    body["leaderboard_today_url"]     = DEFAULT_LEADERBOARD_URL
    body["leaderboard_yesterday_url"] = DEFAULT_LEADERBOARD_YDAY_URL
    body["leaderboard_url"]           = DEFAULT_LEADERBOARD_URL

POST_PATHS: Dict[str, Tuple[Callable[[Dict[str, Any]], None], Callable[[Dict[str, Any]], None]]] = {
    "/evaluate_by_urls_easystreet31":          (_merge_common, _override_common),
    "/scan_by_urls_easystreet31":              (_merge_common, _override_common),
    "/scan_rival_by_urls_easystreet31":        (_merge_common, _override_common),
    "/scan_partner_by_urls_easystreet31":      (_merge_common, _override_common),
    "/review_collection_by_urls_easystreet31": (_merge_common, _override_common),
    "/suggest_give_from_collection_by_urls_easystreet31": (_merge_common, _override_common),

    "/family_evaluate_trade_by_urls":          (_merge_family, _override_family),
    "/collection_review_family_by_urls":       (_merge_family, _override_family),
    "/family_transfer_suggestions_by_urls":    (_merge_family, _override_family),
    "/family_transfer_optimize_by_urls":       (_merge_family, _override_family),

    "/leaderboard_delta_by_urls":              (_merge_delta,  _override_delta),
}

TRADE_ENDPOINTS = {"/evaluate_by_urls_easystreet31", "/family_evaluate_trade_by_urls"}
ALLOWED_FRAG_MODES = {"none", "trade_only", "trade_delta"}  # note: "all" is NOT allowed for trade endpoints

class MergeDefaultsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method.upper() != "POST":
            return await call_next(request)
        pair = POST_PATHS.get(request.url.path)
        if not pair:
            return await call_next(request)

        merge_fn, override_fn = pair
        try:
            raw = await request.body()
            body = {} if not raw else json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                return await call_next(request)
        except Exception:
            return await call_next(request)

        prefer_env = bool(body.get("prefer_env_defaults") is True)
        try:
            if prefer_env:
                override_fn(body)
            else:
                merge_fn(body)
        except Exception:
            pass

        # HARDEN: on trade endpoints, coerce fragility_mode unless explicitly in allowed set
        if request.url.path in TRADE_ENDPOINTS:
            mode = str(body.get("fragility_mode") or "").lower()
            if mode not in ALLOWED_FRAG_MODES:
                body["fragility_mode"] = "trade_delta"

        request.state.merged_body = body
        new_raw = json.dumps(body).encode("utf-8")

        async def receive():
            return {"type": "http.request", "body": new_raw, "more_body": False}
        request._receive = receive  # type: ignore[attr-defined]
        return await call_next(request)

app.add_middleware(MergeDefaultsMiddleware)

# ========= Fragility filter (trade-only, deep, delta-by-default) =========
def _expand_trade_players(body: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for line in body.get("trade", []) or []:
        p = line.get("players")
        if isinstance(p, str) and p.strip():
            names.append(p.strip())
    # split on "/" and ","; keep originals if no splits
    expanded: List[str] = []
    for n in names:
        parts = [s.strip() for s in n.replace(",", "/").split("/") if s.strip()]
        expanded.extend(parts if parts else [n])
    # unique case-insensitive
    seen = set()
    out: List[str] = []
    for n in expanded:
        k = n.lower()
        if k not in seen:
            seen.add(k); out.append(n)
    return out

def _read_margin(d: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    # Accept common field names; strings allowed
    before = d.get("margin_before", d.get("before_margin"))
    after  = d.get("margin_after",  d.get("after_margin", d.get("margin")))
    try:
        b = float(before) if before is not None else None
    except:
        b = None
    try:
        a = float(after) if after is not None else None
    except:
        a = None
    return b, a

def _get_item_name(x: Dict[str, Any]) -> Optional[str]:
    nm = x.get("player") or x.get("name") or x.get("players")
    return nm if isinstance(nm, str) and nm.strip() else None

def _looks_like_frag_list(lst: Any) -> bool:
    # Heuristic: a list of dicts where most have a name, and at least one has margin-ish fields or fragile flag
    if not isinstance(lst, list) or not lst:
        return False
    dicts = [x for x in lst if isinstance(x, dict)]
    if not dicts:
        return False
    namey = [x for x in dicts if _get_item_name(x)]
    if not namey:
        return False
    # margins or flags
    for x in dicts:
        if any(k in x for k in ("margin", "margin_after", "margin_before", "before_margin", "after_margin", "fragile", "is_fragile", "status")):
            return True
    return False

def _keep_item(x: Dict[str, Any], tset: set, defend_buffer: int, mode: str) -> bool:
    nm = _get_item_name(x)
    if not nm or nm.lower() not in tset:
        return False  # only traded players
    b, a = _read_margin(x)
    # If no margins provided, we cannot compute delta — treat as not eligible (hide).
    if a is None and b is None:
        return False
    if mode == "trade_delta":
        if a is None or b is None:
            return (a is not None and a <= defend_buffer)  # fallback to trade_only
        return (b > defend_buffer) and (a <= defend_buffer)
    elif mode == "trade_only":
        return bool(a is not None and a <= defend_buffer)
    elif mode == "none":
        return False
    else:  # defensive default
        return False

def _deep_filter(obj: Any, tset: set, defend_buffer: int, mode: str) -> Any:
    if isinstance(obj, list):
        if _looks_like_frag_list(obj):
            return [x for x in obj if isinstance(x, dict) and _keep_item(x, tset, defend_buffer, mode)]
        else:
            return [_deep_filter(x, tset, defend_buffer, mode) for x in obj]
    elif isinstance(obj, dict):
        newd: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, list) and _looks_like_frag_list(v):
                newd[k] = [x for x in v if isinstance(x, dict) and _keep_item(x, tset, defend_buffer, mode)]
            else:
                newd[k] = _deep_filter(v, tset, defend_buffer, mode)
        return newd
    else:
        return obj

class FragilityFilterMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        try:
            if (request.method.upper() == "POST"
                and request.url.path in {"/evaluate_by_urls_easystreet31", "/family_evaluate_trade_by_urls"}
                and getattr(request, "state", None)
                and isinstance(request.state.merged_body, dict)):

                body = request.state.merged_body
                raw_mode = body.get("fragility_mode")
                mode = str(raw_mode or "trade_delta").lower()
                # enforce allowed modes again here (double safety)
                if mode not in ALLOWED_FRAG_MODES:
                    mode = "trade_delta"

                defend_buffer = int(body.get("defend_buffer") or body.get("defend_buffer_all") or DEFAULT_DEFEND_BUFFER_ALL or 15)
                tset = {p.lower() for p in _expand_trade_players(body)}
                if not tset:
                    return response

                # Read original response body
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
        return response

app.add_middleware(FragilityFilterMiddleware)

# ========= diagnostics =========
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
        "trade_fragility_default": "trade_delta (deep filter on traded players only; 'all' is coerced)"
    }
