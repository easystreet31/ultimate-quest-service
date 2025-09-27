# main.py — wrapper around your core FastAPI app (app_core.py)
# Adds:
#  * Env-default merging (unchanged)
#  * Optional "prefer_env_defaults" override (unchanged)
#  * FragilityFilterMiddleware: for trade eval endpoints, show only trade-created fragility (delta) on traded players.

from __future__ import annotations
import os, json
from typing import Any, Dict, List, Tuple, Callable, Optional

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# ---- import the real app ----
_core_err = None
try:
    from app_core import app as core_app
except Exception as e:
    _core_err = e
    core_app = None

if core_app is None:
    raise RuntimeError(f"Could not import app_core.app: {_core_err!r}")

app: FastAPI = core_app

# ---------- env helpers ----------
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

# ---------- URLs / defaults from env ----------
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

# ---------- request merging / override ----------
def _merge_common(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_url", DEFAULT_LEADERBOARD_URL)
    body.setdefault("holdings_url", DEFAULT_HOLDINGS_E31_URL)
    body.setdefault("collection_url", DEFAULT_POOL_COLLECTION_URL)
    body.setdefault("my_collection_url", DEFAULT_COLLECTION_E31_URL)
    body.setdefault("defend_buffer", DEFAULT_DEFEND_BUFFER_ALL)
    if "target_rivals" not in body and DEFAULT_TARGET_RIVALS:
        body["target_rivals"] = DEFAULT_TARGET_RIVALS
    # default fragility behavior for trade eval (only when caller didn’t specify)
    if body.get("fragility_mode") is None:
        if body.get("trade") and isinstance(body["trade"], list):
            body["fragility_mode"] = "trade_delta"

def _override_common(body: Dict[str, Any]) -> None:
    body["leaderboard_url"] = DEFAULT_LEADERBOARD_URL
    body["holdings_url"] = DEFAULT_HOLDINGS_E31_URL
    body["collection_url"] = DEFAULT_POOL_COLLECTION_URL
    body["my_collection_url"] = DEFAULT_COLLECTION_E31_URL
    body["defend_buffer"] = DEFAULT_DEFEND_BUFFER_ALL
    if DEFAULT_TARGET_RIVALS:
        body["target_rivals"] = DEFAULT_TARGET_RIVALS
    if body.get("trade") and isinstance(body["trade"], list) and body.get("fragility_mode") is None:
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
    if body.get("fragility_mode") is None:
        if body.get("trade") and isinstance(body["trade"], list):
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
    if body.get("trade") and isinstance(body["trade"], list) and body.get("fragility_mode") is None:
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

        request.state.merged_body = body
        new_raw = json.dumps(body).encode("utf-8")

        async def receive():
            return {"type": "http.request", "body": new_raw, "more_body": False}
        request._receive = receive  # type: ignore[attr-defined]
        return await call_next(request)

app.add_middleware(MergeDefaultsMiddleware)

# ---------- Fragility filter (trade-only, delta-by-default) ----------
TRADE_ENDPOINTS = {"/evaluate_by_urls_easystreet31", "/family_evaluate_trade_by_urls"}

def _trade_players_from_body(body: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for line in body.get("trade", []) or []:
        p = line.get("players")
        if isinstance(p, str) and p.strip():
            names.append(p.strip())
    # Split multi-subject like "A/B" into separate names too
    expanded: List[str] = []
    for n in names:
        parts = [s.strip() for s in n.replace(",", "/").split("/") if s.strip()]
        expanded.extend(parts if parts else [n])
    # unique, case-insensitive
    seen = set()
    out: List[str] = []
    for n in expanded:
        k = n.lower()
        if k not in seen:
            seen.add(k); out.append(n)
    return out

def _normalize_frag_list(any_list) -> List[Dict[str, Any]]:
    if not isinstance(any_list, list):
        return []
    out = []
    for item in any_list:
        if isinstance(item, dict):
            out.append(item)
    return out

def _filter_fragility(payload: Dict[str, Any], trade_players: List[str], defend_buffer: int, mode: str) -> Dict[str, Any]:
    # Known fragility keys we may see from core
    CAND_KEYS = [
        "fragility_alerts", "fragile_firsts", "thin_firsts",
        "post_trade_fragility", "fragility", "fragility_after"
    ]
    tset = {p.lower() for p in trade_players}

    def keep_item(d: Dict[str, Any]) -> bool:
        # Extract player name and margins
        nm = d.get("player") or d.get("name") or d.get("players")
        if not isinstance(nm, str):
            return False
        nm_l = nm.lower()
        if nm_l not in tset:
            return False  # only players in the trade
        # margins
        before = d.get("margin_before") or d.get("before_margin")
        after  = d.get("margin_after")  or d.get("after_margin") or d.get("margin")
        try:
            after_val = float(after) if after is not None else None
            before_val = float(before) if before is not None else None
        except:
            after_val = before_val = None

        if mode == "trade_delta":
            if after_val is None or before_val is None:
                # cannot compute delta; drop to trade_only behavior
                return bool(after_val is not None and after_val <= defend_buffer)
            return (before_val > defend_buffer) and (after_val <= defend_buffer)
        elif mode == "trade_only":
            return bool(after_val is not None and after_val <= defend_buffer)
        elif mode == "none":
            return False
        else:  # "all"
            return True

    # Apply filter
    found_any = False
    for k in CAND_KEYS:
        if k in payload:
            lst = _normalize_frag_list(payload.get(k))
            if mode == "none":
                payload[k] = []
                found_any = True
                continue
            filtered = [d for d in lst if keep_item(d)]
            payload[k] = filtered
            found_any = True

    # If core used some other key, leave it as-is (we won't mutate unknown shapes)
    # Add a marker so you know filter ran
    payload.setdefault("_notes", [])
    payload["_notes"].append({"fragility_filter": mode, "trade_players": trade_players})

    # If we removed all fragility items, also surface a clean note for the UI
    if found_any:
        # check if every known list is empty
        empties = True
        for k in CAND_KEYS:
            if isinstance(payload.get(k), list) and len(payload.get(k)) > 0:
                empties = False; break
        if empties:
            payload["_notes"].append({"fragility_result": "suppressed_or_none"})
    return payload

class FragilityFilterMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        try:
            if (request.method.upper() == "POST" and
                request.url.path in TRADE_ENDPOINTS and
                getattr(request, "state", None) and
                isinstance(request.state.merged_body, dict)):

                body = request.state.merged_body
                mode = str(body.get("fragility_mode") or "trade_delta").lower()
                # pull defend buffer: single-account "defend_buffer" or family "defend_buffer_all"
                defend_buffer = int(body.get("defend_buffer") or body.get("defend_buffer_all") or DEFAULT_DEFEND_BUFFER_ALL or 15)
                trade_players = _trade_players_from_body(body)

                # Only attempt to mutate JSON responses
                raw = b""
                if hasattr(response, "body_iterator"):
                    # Read existing body
                    chunks = []
                    async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                else:
                    raw = getattr(response, "body", b"") or b""

                # decode and filter
                if raw:
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                        if isinstance(payload, dict) and trade_players:
                            new_payload = _filter_fragility(payload, trade_players, defend_buffer, mode)
                            new_raw = json.dumps(new_payload, ensure_ascii=False).encode("utf-8")
                            response = Response(content=new_raw, status_code=response.status_code, media_type="application/json")
                        else:
                            # not a dict or no trade players; leave as is
                            response = Response(content=raw, status_code=response.status_code, media_type="application/json")
                    except Exception:
                        # not JSON → return raw
                        response = Response(content=raw, status_code=response.status_code, media_type=response.media_type)
                # else: no body; return as is
        except Exception:
            # On any error, do not block the response
            return response
        return response

app.add_middleware(FragilityFilterMiddleware)

# ---------- Diagnostics ----------
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
        "trade_fragility_default": "trade_delta (only show trade-created fragility on traded players)"
    }
