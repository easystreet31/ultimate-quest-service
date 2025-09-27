# baseline_gets_filter.py
# Middleware that removes collection picks which duplicate baseline GETs from a trade.
# - Excludes exact matches by (players set, SP)
# - Excludes exact matches by card_number (if present)
# - Only activates when:
#     a) request JSON has {"exclude_baseline_gets": true, "baseline_trade":[...]}
#     b) path is a collection-review endpoint

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import json
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

# Endpoints whose JSON we will post-filter
_FILTER_PATHS = {
    "/review_collection_by_urls_easystreet31",
    "/collection_review_family_by_urls",
}

# Keys we may need to filter in the JSON payloads (arrays of pick dicts)
_PICK_LIST_KEYS = [
    "qp_positive_picks",
    "shore_thin_from_collection",
    "take_first_from_collection",
    "enter_top3_from_collection",
    "rival_priority_picks",
    "family_picks"
]

# -----------------------------
# Helpers to normalize/compare
# -----------------------------

def _norm_players_key(players_value: Any) -> Optional[str]:
    """
    Accepts either a string 'A/B/C' or a list of names.
    Returns canonical 'A|B|C' (sorted, stripped) or None.
    """
    if players_value is None:
        return None
    if isinstance(players_value, (list, tuple, set)):
        parts = [str(p).strip() for p in players_value if str(p).strip()]
    else:
        raw = str(players_value)
        for sep in ['&', ',']:
            raw = raw.replace(sep, '/')
        parts = [p.strip() for p in raw.split('/') if p.strip()]
    if not parts:
        return None
    return '|'.join(sorted(parts))


def _make_trade_get_keys(baseline_trade: Iterable[Dict[str, Any]]) -> Tuple[Set[Tuple[str, int]], Set[str]]:
    """
    Build exclusion keys from baseline trade GET lines.

    Two flavors:
      • players/SP key:   ( 'A|B|C', sp )
      • card_number key:  'UI-87'   (uppercased)
    """
    keys_players_sp: Set[Tuple[str, int]] = set()
    keys_cardnum: Set[str] = set()
    if not baseline_trade:
        return keys_players_sp, keys_cardnum

    for line in baseline_trade:
        if not isinstance(line, dict):
            continue
        if str(line.get("side", "")).upper() != "GET":
            continue
        pk = _norm_players_key(line.get("players"))
        sp_raw = line.get("sp")
        sp_val = None
        if sp_raw is not None:
            try:
                sp_val = int(float(sp_raw))
            except Exception:
                sp_val = None
        if pk and sp_val is not None:
            keys_players_sp.add((pk, sp_val))

        card_no = line.get("card_number")
        if card_no:
            keys_cardnum.add(str(card_no).strip().upper())

    return keys_players_sp, keys_cardnum


def _pick_matches_exclusion(pick: Dict[str, Any],
                            keys_players_sp: Set[Tuple[str, int]],
                            keys_cardnum: Set[str]) -> bool:
    """
    True if a collection pick collides with baseline GETs by (players, SP) or card_number.
    """
    # players can be string or list
    players_val = pick.get("players") or pick.get("players_list")
    pk = _norm_players_key(players_val)

    sp_val = pick.get("sp", pick.get("subject_points"))
    sp = None
    if sp_val is not None:
        try:
            sp = int(float(sp_val))
        except Exception:
            sp = None

    card_no = pick.get("card_number")
    card_no_up = str(card_no).strip().upper() if card_no else None

    if card_no_up and card_no_up in keys_cardnum:
        return True
    if pk and (sp is not None) and (pk, sp) in keys_players_sp:
        return True
    return False


def _filter_picks_excluding_baseline_gets(picks: List[Dict[str, Any]],
                                          baseline_trade: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter a list of pick dicts, removing items that match the baseline GETs.
    """
    if not picks or not baseline_trade:
        return picks or []

    keys_players_sp, keys_cardnum = _make_trade_get_keys(baseline_trade)
    if not keys_players_sp and not keys_cardnum:
        return picks

    filtered: List[Dict[str, Any]] = []
    for p in picks:
        try:
            if not _pick_matches_exclusion(p, keys_players_sp, keys_cardnum):
                filtered.append(p)
        except Exception:
            # If malformed, keep the pick (fail-open)
            filtered.append(p)
    return filtered


# -----------------------------
# Middleware
# -----------------------------

class BaselineGetsFilterMiddleware(BaseHTTPMiddleware):
    """
    Intercepts POST JSON requests to collection-review endpoints.
    If payload has {"exclude_baseline_gets": true, "baseline_trade": [...]},
    removes any recommended picks that collide with the baseline GETs.
    """

    async def dispatch(self, request: Request, call_next):
        # Only watch certain endpoints + POST
        if request.method != "POST" or request.url.path not in _FILTER_PATHS:
            return await call_next(request)

        # Read request body once and re-attach it so downstream handlers still receive it
        try:
            body_bytes = await request.body()
        except Exception:
            body_bytes = b""
        try:
            req_json = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            req_json = {}

        # Re-supply the same body to the downstream app
        async def _receive():
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        request._receive = _receive  # type: ignore[attr-defined]

        # Run the endpoint
        response = await call_next(request)

        # Filter only JSON responses
        ctype = (response.headers.get("content-type") or "").lower()
        if "application/json" not in ctype:
            return response

        # Only activate if the request asked for it and gave a baseline trade
        if not (req_json.get("exclude_baseline_gets") and req_json.get("baseline_trade")):
            return response

        # Collect response body
        resp_bytes = b""
        async for chunk in response.body_iterator:
            resp_bytes += chunk

        try:
            payload = json.loads(resp_bytes.decode("utf-8")) if resp_bytes else {}
        except Exception:
            # If the response isn't JSON parseable, pass through unchanged
            return Response(content=resp_bytes, status_code=response.status_code, headers=dict(response.headers))

        # Filter all known pick lists if present
        for key in _ PICK_LIST_KEYS:
            if isinstance(payload.get(key), list):
                payload[key] = _filter_picks_excluding_baseline_gets(payload[key], req_json["baseline_trade"])

        # Return the modified JSON
        return JSONResponse(payload, status_code=response.status_code, headers=dict(response.headers))
