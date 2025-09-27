# baseline_gets_filter.py — middleware to avoid “double-dipping” on players already in the trade GET lines
from __future__ import annotations
import json
from typing import Any, Dict, List, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse, JSONResponse
from starlette.types import ASGIApp

def _safe_json_loads(b: bytes) -> Any | None:
    try:
        return json.loads(b.decode("utf-8"))
    except Exception:
        return None

def _collect_trade_get_players(payload: Any) -> Set[str]:
    """
    Extract player names from trade GET lines in a request body.
    Supports shapes like:
      {"trade":[{"side":"GET","players":"A/B","sp":2}, ...]}
      {"family_trade":{"trade":[...]}}
    Returns a set of individual player names (split on '/' and trimmed).
    """
    names: Set[str] = set()
    if not isinstance(payload, dict):
        return names

    # most common shapes
    trade = None
    if "trade" in payload and isinstance(payload["trade"], list):
        trade = payload["trade"]
    elif "family_trade" in payload and isinstance(payload["family_trade"], dict):
        ft = payload["family_trade"]
        if "trade" in ft and isinstance(ft["trade"], list):
            trade = ft["trade"]

    if not trade:
        return names

    for line in trade:
        try:
            if isinstance(line, dict) and str(line.get("side", "")).upper() == "GET":
                players_field = str(line.get("players", "")).strip()
                if not players_field:
                    continue
                # split multi-subject names by "/" and normalize
                parts = [p.strip() for p in players_field.split("/") if p.strip()]
                for p in parts:
                    names.add(p)
        except Exception:
            # be lenient
            continue
    return names

def _filter_counter_picks(body: Any, trade_get_players: Set[str]) -> Any:
    """
    Given a response JSON body and the set of trade GET players,
    remove counter picks that duplicate those players.

    Expected tolerant shapes:
      {
        "counter": {
          "picks": [
             {"players": ["Brock Boeser"], "card_name": "...", ...},
             {"players": ["John Tavares"], ...},
             ...
          ],
          "omitted": N,
          ...
        },
        ...
      }

    If the shape is different, return the body unchanged.
    """
    if not trade_get_players:
        return body
    if not isinstance(body, dict):
        return body

    counter = body.get("counter")
    if not isinstance(counter, dict):
        return body

    picks = counter.get("picks")
    if not isinstance(picks, list):
        return body

    # Filter
    filtered: List[Dict[str, Any]] = []
    removed = 0
    for pick in picks:
        try:
            players = pick.get("players")
            # players may be a list ["Name", "Name"] or a single string "Name/Name"
            if isinstance(players, list):
                pl = [str(x).strip() for x in players if str(x).strip()]
            else:
                pl = [s.strip() for s in str(players or "").split("/") if s.strip()]

            # if any of the players on this pick intersect the trade GET set, drop it
            if any(p in trade_get_players for p in pl):
                removed += 1
                continue

            filtered.append(pick)
        except Exception:
            # Be permissive; if unreadable, keep it rather than risk hiding info
            filtered.append(pick)

    counter["picks"] = filtered
    # best-effort: bump an omitted count if present
    if "omitted" in counter and isinstance(counter["omitted"], int):
        counter["omitted"] += removed
    else:
        counter["omitted"] = removed

    body["counter"] = counter
    return body

class BaselineGetsFilterMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware:
    - Reads the request body to collect trade GET players (if present).
    - Lets the app handle the request.
    - If the response is JSON and contains a “counter” block (family trade + counter),
      removes counter picks that duplicate players already GET in the trade.
    - For all other requests/shapes, passes through unchanged.

    NOTE: This is non-invasive. If the request/response shapes differ from the
    expectations above, nothing is modified.
    """

    async def dispatch(self, request: Request, call_next):
        # 1) Peek at request body to collect trade GET players (best-effort)
        try:
            raw = await request.body()
        except Exception:
            raw = b""

        trade_get_players = _collect_trade_get_players(_safe_json_loads(raw))

        # Rebuild the request stream for downstream (so FastAPI can read it again)
        async def receive():
            return {"type": "http.request", "body": raw, "more_body": False}

        request._receive = receive  # type: ignore[attr-defined]

        # 2) Call downstream
        response: Response = await call_next(request)

        # 3) Only attempt to filter JSON responses with a body
        try:
            # If not JSON, pass through
            if "application/json" not in (response.headers.get("content-type") or ""):
                return response

            # Read the existing body
            if isinstance(response, (JSONResponse,)):
                body_obj = response.body
            else:
                body_bytes = b""
                async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                    body_bytes += chunk
                body_obj = body_bytes

            body_json = _safe_json_loads(body_obj if isinstance(body_obj, (bytes, bytearray)) else bytes(body_obj))
            if body_json is None:
                # Not JSON decodable → pass through
                return response

            # 4) Filter counter picks if applicable
            new_body = _filter_counter_picks(body_json, trade_get_players)
            # 5) Return a fresh JSON response
            return JSONResponse(new_body, status_code=getattr(response, "status_code", 200), headers=dict(response.headers))

        except Exception:
            # Any failure → return the original response
            return response
