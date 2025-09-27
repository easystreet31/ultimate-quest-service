# main.py — robust defaults + strict trade fragility filter
# Version: 4.0.1-tradefrag-hard

from __future__ import annotations
import os, json, re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# ---- import the real app (your domain logic) ----
_core_err = None
try:
    from app_core import app as core_app
except Exception as e:
    _core_err = e
    core_app = None

if core_app is None:
    raise RuntimeError(f"Could not import app_core.app: {_core_err!r}")

app: FastAPI = core_app

# ---- helpers ----
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

# ---- environment defaults ----
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

# ---- endpoint groups ----
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

TRADE_ENDPOINTS = {
    "/evaluate_by_urls_easystreet31",
    "/family_evaluate_trade_by_urls",
}

# ---- placeholders we scrub ----
PLACEHOLDER_TOKENS = {
    "E31_default", "DC_default", "FE_default", "LB_default",
    "POOL_default", "E31", "DC", "FE", "default_url"
}

def _looks_bad_url(v: Any) -> bool:
    if not isinstance(v, str):
        return True
    s = v.strip()
    if not s:
        return True
    if any(tok.lower() in s.lower() for tok in PLACEHOLDER_TOKENS):
        return True
    if not s.startswith("http"):
        return True
    return False

def _sanitize_common_urls(body: Dict[str, Any]) -> None:
    # single-account defaults if absent or placeholders
    if _looks_bad_url(body.get("leaderboard_url")) and DEFAULT_LEADERBOARD_URL:
        body["leaderboard_url"] = DEFAULT_LEADERBOARD_URL
    if _looks_bad_url(body.get("holdings_url")) and DEFAULT_HOLDINGS_E31_URL:
        body["holdings_url"] = DEFAULT_HOLDINGS_E31_URL
    if _looks_bad_url(body.get("collection_url")) and DEFAULT_POOL_COLLECTION_URL:
        body["collection_url"] = DEFAULT_POOL_COLLECTION_URL
    if _looks_bad_url(body.get("my_collection_url")) and DEFAULT_COLLECTION_E31_URL:
        body["my_collection_url"] = DEFAULT_COLLECTION_E31_URL

def _force_family_urls(body: Dict[str, Any]) -> None:
    # always force env defaults for family endpoints (escape hatch below)
    body["leaderboard_url"]    = DEFAULT_LEADERBOARD_URL
    body["holdings_e31_url"]   = DEFAULT_HOLDINGS_E31_URL
    body["holdings_dc_url"]    = DEFAULT_HOLDINGS_DC_URL
    body["holdings_fe_url"]    = DEFAULT_HOLDINGS_FE_URL
    body["collection_e31_url"] = DEFAULT_COLLECTION_E31_URL
    body["collection_dc_url"]  = DEFAULT_COLLECTION_DC_URL
    body["collection_fe_url"]  = DEFAULT_COLLECTION_FE_URL
    if "defend_buffer_all" not in body:
        body["defend_buffer_all"] = DEFAULT_DEFEND_BUFFER_ALL

def _merge_family_defaults(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_url",    DEFAULT_LEADERBOARD_URL)
    body.setdefault("holdings_e31_url",   DEFAULT_HOLDINGS_E31_URL)
    body.setdefault("holdings_dc_url",    DEFAULT_HOLDINGS_DC_URL)
    body.setdefault("holdings_fe_url",    DEFAULT_HOLDINGS_FE_URL)
    body.setdefault("collection_e31_url", DEFAULT_COLLECTION_E31_URL)
    body.setdefault("collection_dc_url",  DEFAULT_COLLECTION_DC_URL)
    body.setdefault("collection_fe_url",  DEFAULT_COLLECTION_FE_URL)
    body.setdefault("defend_buffer_all",  DEFAULT_DEFEND_BUFFER_ALL)

def _merge_single_defaults(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_url",    DEFAULT_LEADERBOARD_URL)
    body.setdefault("holdings_url",       DEFAULT_HOLDINGS_E31_URL)
    body.setdefault("collection_url",     DEFAULT_POOL_COLLECTION_URL)
    body.setdefault("my_collection_url",  DEFAULT_COLLECTION_E31_URL)
    body.setdefault("defend_buffer",      DEFAULT_DEFEND_BUFFER_ALL)
    if "target_rivals" not in body and DEFAULT_TARGET_RIVALS:
        body["target_rivals"] = DEFAULT_TARGET_RIVALS

def _merge_delta_defaults(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_today_url",     DEFAULT_LEADERBOARD_URL)
    body.setdefault("leaderboard_yesterday_url", DEFAULT_LEADERBOARD_YDAY_URL)
    body.setdefault("leaderboard_url", body.get("leaderboard_today_url") or DEFAULT_LEADERBOARD_URL)

# ---- middleware #1: merge/override defaults ----
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

        # Apply defaults/overrides
        if path in FAMILY_PATHS:
            # escape hatch: allow_custom_urls = true
            if body.get("allow_custom_urls") is True:
                _merge_family_defaults(body)
            else:
                _force_family_urls(body)
        elif path in SINGLE_PATHS:
            _merge_single_defaults(body)
            _sanitize_common_urls(body)
        elif path in DELTA_PATHS:
            _merge_delta_defaults(body)

        # default fragility mode for trade endpoints
        if path in TRADE_ENDPOINTS and body.get("fragility_mode") is None:
            body["fragility_mode"] = "trade_delta"  # strict by default

        # stash merged body for downstream middlewares
        request.state.merged_body = body
        new_raw = json.dumps(body, ensure_ascii=False).encode("utf-8")

        async def receive():
            return {"type": "http.request", "body": new_raw, "more_body": False}
        request._receive = receive  # type: ignore[attr-defined]

        return await call_next(request)

app.add_middleware(MergeDefaultsMiddleware)

# ---- trade fragility filter (middleware #2) ----

ALLOWED_FRAG_MODES = {"none", "trade_only", "trade_delta"}

NAME_TOKENS_SPLIT = re.compile(r"[\/,&]")

def _expand_trade_players(body: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for line in body.get("trade") or []:
        players = line.get("players")
        if not isinstance(players, str):
            continue
        # Split on '/', ',', '&' and also allow ' / ' variants
        parts = [s.strip() for s in NAME_TOKENS_SPLIT.split(players) if s.strip()]
        if parts:
            out.extend(parts)
        else:
            out.append(players.strip())
    # uniquify case-insensitive
    seen, uniq = set(), []
    for p in out:
        k = p.lower()
        if k not in seen:
            seen.add(k); uniq.append(p)
    return uniq

def _parse_name_from_string(s: str) -> Optional[str]:
    # Try formats like:
    #   "Evander Kane (margin 1)"
    #   "Evander Kane — margin 1"
    #   "Evander Kane → margin 1"
    #   "Evander Kane 1"
    m = re.match(r"^\s*([A-Za-z .'\-]+?)\s*(?:[—\-→(]|$)", s)
    if m:
        name = m.group(1).strip()
        return name if name else None
    # Last resort: take before first digit
    m2 = re.match(r"^\s*([A-Za-z .'\-]+?)\s*\d", s)
    if m2:
        name = m2.group(1).strip()
        return name if name else None
    return None

def _get_name(x: Dict[str, Any]) -> Optional[str]:
    for k in ("player","name","players","subject","title"):
        v = x.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _read_margin_pair(d: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    # we accept several spellings from various endpoints
    b = d.get("margin_before", d.get("before_margin", d.get("before")))
    a = d.get("margin_after",  d.get("after_margin",  d.get("after", d.get("margin"))))
    try: b = float(b) if b is not None else None
    except: b = None
    try: a = float(a) if a is not None else None
    except: a = None
    return b, a

def _keep_dict_item(x: Dict[str, Any], tset: set, buffer_: int, mode: str) -> bool:
    nm = _get_name(x)
    if not nm:
        return False
    if nm.lower() not in tset:
        return False
    if mode == "trade_only":
        a = _read_margin_pair(x)[1]
        return (a is not None) and (a <= buffer_)
    if mode == "trade_delta":
        b, a = _read_margin_pair(x)
        if a is None or b is None:
            # degrade to trade_only if we can't see the delta
            return (a is not None) and (a <= buffer_)
        return (b > buffer_) and (a <= buffer_)
    return False

def _keep_string_item(s: str, tset: set, buffer_: int, mode: str) -> bool:
    nm = _parse_name_from_string(s)
    if not nm or nm.lower() not in tset:
        return False
    # If margins aren't parseable from strings, for safety:
    #   trade_delta -> require it to look like a drop (contains "margin 0/1/2/..<=buffer")
    if mode == "trade_delta":
        # accept if string mentions margin <= buffer (e.g., "(margin 0|1|2|…buffer)")
        m = re.search(r"margin\s*([0-9]+)", s, flags=re.I)
        if m:
            try:
                a = int(m.group(1))
                return a <= buffer_
            except:
                return False
        # no numeric margin; conservative: reject
        return False
    if mode == "trade_only":
        m = re.search(r"margin\s*([0-9]+)", s, flags=re.I)
        if m:
            try:
                a = int(m.group(1))
                return a <= buffer_
            except:
                return False
        # if no explicit margin, keep since name is in trade set (best-effort)
        return True
    return False

# keys likely carrying fragility/alerts
SUSPECT_KEYS = re.compile(r"(fragil|thin|alert|defen|firsts)", re.I)

def _filter_fragility_payload(obj: Any, tset: set, buffer_: int, mode: str) -> Any:
    # Deep-walk: filter lists of dicts/strings under fragility-ish keys;
    # leave all other content untouched.
    if isinstance(obj, dict):
        newd: Dict[str, Any] = {}
        for k, v in obj.items():
            if SUSPECT_KEYS.search(k):
                # aggressively filter
                newd[k] = _filter_fragility_node(v, tset, buffer_, mode)
            else:
                newd[k] = _filter_fragility_payload(v, tset, buffer_, mode)
        return newd
    if isinstance(obj, list):
        return [_filter_fragility_payload(x, tset, buffer_, mode) for x in obj]
    return obj

def _filter_fragility_node(v: Any, tset: set, buffer_: int, mode: str) -> Any:
    # v may be:
    #  - list[dict], list[str], dict(account->list), or anything else
    if isinstance(v, list):
        if v and isinstance(v[0], dict):
            kept = [x for x in v if isinstance(x, dict) and _keep_dict_item(x, tset, buffer_, mode)]
            return kept
        if v and isinstance(v[0], str):
            kept = [s for s in v if isinstance(s, str) and _keep_string_item(s, tset, buffer_, mode)]
            return kept
        # mixed or empty
        return []
    if isinstance(v, dict):
        # assume account -> list (or nested dicts)
        nd: Dict[str, Any] = {}
        for ak, av in v.items():
            if isinstance(av, list):
                if av and isinstance(av[0], dict):
                    kept = [x for x in av if isinstance(x, dict) and _keep_dict_item(x, tset, buffer_, mode)]
                    nd[ak] = kept
                elif av and isinstance(av[0], str):
                    kept = [s for s in av if isinstance(s, str) and _keep_string_item(s, tset, buffer_, mode)]
                    nd[ak] = kept
                else:
                    nd[ak] = []
            else:
                nd[ak] = _filter_fragility_node(av, tset, buffer_, mode)
        return nd
    # scalar: drop
    return None

class FragilityFilterMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        try:
            if request.method.upper() != "POST":
                return response
            path = request.url.path
            if path not in TRADE_ENDPOINTS:
                return response
            if not getattr(request, "state", None) or not isinstance(request.state.merged_body, dict):
                return response

            body = request.state.merged_body
            mode = str(body.get("fragility_mode") or "trade_delta").lower()
            if mode not in ALLOWED_FRAG_MODES:
                mode = "trade_delta"

            defend_buffer = int(body.get("defend_buffer")
                                or body.get("defend_buffer_all")
                                or DEFAULT_DEFEND_BUFFER_ALL
                                or 15)

            trade_player_set = {p.lower() for p in _expand_trade_players(body)}
            if not trade_player_set:
                return response

            # read original body
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
                # non-JSON; just return
                return Response(content=raw, status_code=response.status_code, media_type=response.media_type)

            if not isinstance(payload, dict):
                return Response(content=raw, status_code=response.status_code, media_type="application/json")

            # filter in place
            filtered = _filter_fragility_payload(payload, trade_player_set, defend_buffer, mode)

            # annotate for traceability
            notes = filtered.setdefault("_notes", [])
            if isinstance(notes, list):
                notes.append({
                    "fragility_filter": mode,
                    "trade_players": sorted(list({p.title() for p in trade_player_set})),
                    "defend_buffer_used": defend_buffer
                })

            new_raw = json.dumps(filtered, ensure_ascii=False).encode("utf-8")
            return Response(content=new_raw, status_code=response.status_code, media_type="application/json")
        except Exception:
            # on any failure, return original response
            return response

app.add_middleware(FragilityFilterMiddleware)

# ---- diagnostics ----
@app.get("/defaults")
def defaults():
    return {
        "leaderboard_url":           DEFAULT_LEADERBOARD_URL,
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
        "trade_fragility_default": "trade_delta (strict; traded players only)",
        "force_family_urls": True,
        "version": "4.0.1-tradefrag-hard"
    }
