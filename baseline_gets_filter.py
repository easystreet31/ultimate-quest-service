# baseline_gets_filter.py — request body fixer for trade/counter flows
from __future__ import annotations
import json
from typing import Any, Dict, List, Callable, Awaitable
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_TRADE_ENDPOINTS = {
    "/evaluate_by_urls_easystreet31",
    "/family_evaluate_trade_by_urls",
}
_COUNTER_ENDPOINTS = {
    "/family_eval_and_counter_by_urls",
    "/review_collection_by_urls_easystreet31",
    "/family_review_collection_by_urls",
}

def _lower(s: str) -> str:
    return (s or "").strip().lower()

def _extract_get_players(payload: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    trade = payload.get("trade") or payload.get("baseline_trade")
    if isinstance(trade, list):
        for line in trade:
            try:
                if _lower(line.get("side")) == "get":
                    players = line.get("players")
                    if not players:
                        continue
                    # support multi-subject "A/B/C"
                    for p in [x.strip() for x in str(players).split("/")]:
                        if p and p not in out:
                            out.append(p)
            except Exception:
                continue
    return out

class BaselineGetsFilterMiddleware(BaseHTTPMiddleware):
    """
    - Injects fragility_mode='trade_delta' on trade endpoints if missing
    - Adds skip_counter_players on counter endpoints using GET players from trade
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Any]]):
        if request.method != "POST":
            return await call_next(request)

        path = request.url.path
        if path not in (_TRADE_ENDPOINTS | _COUNTER_ENDPOINTS):
            return await call_next(request)

        # Read the original body
        try:
            raw = await request.body()
            if not raw:
                return await call_next(request)
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                return await call_next(request)
        except Exception:
            return await call_next(request)

        changed = False

        # 1) Trade-only fragility (default)
        if path in _TRADE_ENDPOINTS and "fragility_mode" not in data:
            data["fragility_mode"] = "trade_delta"
            changed = True

        # 2) Exclude GET players from counter collections
        if path in _COUNTER_ENDPOINTS:
            skip = set(data.get("skip_counter_players") or [])
            # prefer baseline_trade if present; fall back to trade
            get_players = _extract_get_players({"trade": data.get("trade")}) or \
                          _extract_get_players({"baseline_trade": data.get("baseline_trade")})
            for p in get_players:
                skip.add(p)
            if skip:
                data["skip_counter_players"] = sorted(skip)
                changed = True

        if not changed:
            return await call_next(request)

        # Re-inject modified body
        new_body = json.dumps(data).encode("utf-8")

        async def receive() -> dict:
            return {"type": "http.request", "body": new_body, "more_body": False}

        request._receive = receive  # type: ignore[attr-defined]
        return await call_next(request)
