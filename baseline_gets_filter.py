# baseline_gets_filter.py — middleware to avoid suggesting counter buys
# for players already present in the trade's GET lines.
#
# Behavior:
#  - Reads the request JSON to collect the set of "GET" players (splitting multi-subject "A/B" by "/").
#  - Lets the app run.
#  - If the JSON response contains a "counter" block with "picks": [...],
#    remove any pick whose players intersect the GET players from the request.
#  - If shapes differ, the middleware is a safe no-op.

from __future__ import annotations
import json
from typing import Any, Dict, List, Set, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.types import ASGIApp

def _safe_json_loads(b: bytes) -> Optional[Any]:
    try:
        return json.loads(b.decode("utf-8"))
    except Exception:
        return None

def _collect_trade_get_players(payload: Any) -> Set[str]:
    """
    Extract player names from trade GET lines in request payload.

    Supported shapes:
      { "trade": [{ "side":"GET", "players":"A/B", "sp":2 }, ...] }
      { "family_trade": { "trade": [ ... ] } }

    Returns a set of individual player names, split on '/' and stripped.
    """
    names: Set[str] = set()
    if not isinstance(payload, dict):
        return names

    trade = None
    if isinstance(payload.get("trade"), list):
        trade = payload["trade"]
    else:
        ft = payload.get("family_trade")
        if isinstance(ft, dict) and isinstance(ft.get("trade"), list):
            trade = ft["trade"]

    if not trade:
        return names

    for line in trade:
        if not isinstance(line, dict):
            continue
        side = str(line.get("side", "")).strip().upper()
        if side != "GET":
            continue
        players_field = str(line.get("players", "")).strip()
        if not players_field:
            continue
        for p in players_field.split("/"):
            p = p.strip()
            if p:
                names.add(p)
    return names

def _normalize_pick_players(pick: Dict[str, Any]) -> List[str]:
    """
    Read players from a single counter pick.
    Accepts either:
      - "players": ["Name", "Name2"]
      - "players": "Name/Name2"
      - or sometimes "card_players": [...]
    Returns a list of normalized player names (stripped).
    """
    raw = pick.get("players")
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [s.strip() for s in raw.split("/") if s.strip()]
    # fallback alternate field some planners use
    raw2 = pick.get("card_players")
    if isinstance(raw2, list):
        return [str(x).strip() for x in raw2 if str(x).strip()]
    return []

def _filter_counter_picks(body: Any, trade_get_players: Set[str]) -> Any:
    """
    Given a JSON response body and the set of GET players from the request,
    remove counter picks that duplicate those players.

    Expected tolerant shape:
      {
        "counter": {
           "picks": [ { "players": [... or "A/B" ...], ... }, ... ],
           "omitted": <int>?, ...
        },
        ...
      }

    If the shape doesn't match, returns the original body unchanged.
    """
    if not trade_get_players or not isinstance(body, dict):
        return body

    counter = body.get("counter")
    if not isinstance(counter, dict):
        return body

    picks = counter.get("picks")
    if not isinstance(picks, list):
        return body

    filtered: List[Dict[str, Any]] = []
    removed = 0

    for pick in picks:
        try:
            players = _normalize_pick_players(pick)
            # If any overlap with GET players, drop this pick
            if any(p in trade_get_players for p in players):
                removed += 1
                continue
            filtered.append(pick)
        except Exception:
            # Be permissive — if anything goes wrong, keep the pick
            filtered.append(pick)

    counter["picks"] = filtered
    # keep or set 'omitted' to reflect removals
    try:
        prev = int(counter.get("omitted", 0))
    except Exception:
        prev = 0
    counter["omitted"] = prev + removed

    body["counter"] = counter
    return body

class BaselineGetsFilterMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware:
      1) Read request body once → record GET players (best-effort).
      2) Call downstream app.
      3) If response is JSON and contains "counter.picks", remove picks for players already in GETs.
      4) Otherwise pass through unchanged.

    It is deliberately conservative — if shapes differ, it does nothing.
    """

    async def dispatch(self, request: Request, call_next):
        # --- 1) Read the request body (so downstream can still read it later) ---
        try:
            raw_req = await request.body()
        except Exception:
            raw_req = b""

        # Collect GET players from request (if any)
        trade_get_players = _collect_trade_get_players(_safe_json_loads(raw_req))

        # Re-supply the original body to downstream handlers
        async def _receive():
            return {"type": "http.request", "body": raw_req, "more_body": False}
        request._receive = _receive  # type: ignore[attr-defined]

        # --- 2) Call downstream app ---
        response: Response = await call_next(request)

        # --- 3) Only post-process JSON responses ---
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" not in content_type:
            return response

        # Get the original JSON body safely
        body_bytes: Optional[bytes] = None
        try:
            # JSONResponse has .body already as bytes
            if isinstance(response, JSONResponse):
                body_bytes = response.body
            else:
                # Starlette Response often has .body; if not, we won't touch it
                body_attr = getattr(response, "body", None)
                if isinstance(body_attr, (bytes, bytearray)):
                    body_bytes = bytes(body_attr)
        except Exception:
            body_bytes = None

        if body_bytes is None:
            # Could be a streaming response — do not consume it; pass through
            return response

        body_json = _safe_json_loads(body_bytes)
        if body_json is None:
            return response

        # Filter counter picks if applicable
        new_body = _filter_counter_picks(body_json, trade_get_players)

        # Rebuild a JSONResponse; copy headers except content-length (it'll be re-set)
        new_headers = dict(response.headers)
        new_headers.pop("content-length", None)
        return JSONResponse(new_body, status_code=getattr(response, "status_code", 200), headers=new_headers)
