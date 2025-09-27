# baseline_gets_filter.py — middleware to avoid suggesting counter buys
# for players already present in the trade's GET lines.
#
# Behavior:
#   • Reads the request JSON (non-streaming) to collect the set of GET players.
#   • Lets the app run.
#   • If the JSON response contains a "counter": {"picks":[...]}, drops any pick whose players overlap GET.
#   • If shapes differ, it's a safe no‑op.

from __future__ import annotations
import json
from typing import Any, Dict, List, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.types import ASGIApp


def _split_players(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    # split on common delimiters: '/', '&', '+', ',', ' and ', '|', em/en dashes
    import re
    raw = [p.strip() for p in re.split(r"\s*(?:/|&|\+|,| and |\||—|–)\s*", text) if p and p.strip()]
    return list(dict.fromkeys(raw))  # unique-preserving order


def _extract_get_players_from_request(body_bytes: bytes) -> Set[str]:
    try:
        data = json.loads(body_bytes.decode("utf-8"))
    except Exception:
        return set()
    trade = data.get("trade")
    if not isinstance(trade, list):
        return set()
    acc: Set[str] = set()
    for line in trade:
        try:
            if isinstance(line, dict) and (line.get("side") or "").upper() == "GET":
                players = line.get("players", "")
                for p in _split_players(players):
                    acc.add(p)
        except Exception:
            continue
    return acc


def _normalize_pick_players(pick: Dict[str, Any]) -> List[str]:
    # Accept either list ["A","B"] or string "A/B"
    players = pick.get("players")
    if isinstance(players, list):
        return [str(p).strip() for p in players if str(p).strip()]
    if isinstance(players, str):
        return _split_players(players)
    return []


def _filter_counter_picks(body: Any, trade_get_players: Set[str]) -> Any:
    try:
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
        filtered: List[Dict[str, Any]] = []
        removed = 0
        for pick in picks:
            try:
                players = _normalize_pick_players(pick)
                if any(p in trade_get_players for p in players):
                    removed += 1
                    continue
                filtered.append(pick)
            except Exception:
                filtered.append(pick)
        counter["picks"] = filtered
        try:
            prev = int(counter.get("omitted", 0))
        except Exception:
            prev = 0
        counter["omitted"] = prev + removed
        body["counter"] = counter
        return body
    except Exception:
        return body


class BaselineGetsFilterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Read and preserve body for downstream
        try:
            raw_body = await request.body()
        except Exception:
            raw_body = b""
        trade_get_players = _extract_get_players_from_request(raw_body)

        async def receive():
            return {"type": "http.request", "body": raw_body, "more_body": False}

        request = Request(request.scope, receive)

        response = await call_next(request)

        # Only attempt to filter JSON responses with an already-built body
        try:
            body_bytes = getattr(response, "body", None)
            if body_bytes is None:
                return response  # streaming or special response
            try:
                body_json = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                return response
            new_body = _filter_counter_picks(body_json, trade_get_players)
            new_headers = dict(response.headers)
            new_headers.pop("content-length", None)
            return JSONResponse(new_body,
                                status_code=getattr(response, "status_code", 200),
                                headers=new_headers)
        except Exception:
            return response
