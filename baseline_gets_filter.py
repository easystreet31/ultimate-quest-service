# baseline_gets_filter.py — request filter for "trade-only" fragility
# Safe: if payloads don’t match assumptions, it no-ops.
from __future__ import annotations
import json
import re
from typing import Iterable, List, Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_TRADE_PATHS = {
    "/family_evaluate_trade_by_urls",
    "/family_eval_and_counter_by_urls",
    "/evaluate_by_urls_easystreet31",      # harmless here; mostly the family endpoints matter
}

def _split_players(s: str) -> List[str]:
    # Handles "A/B/C" and trims whitespace; keeps distinct names.
    parts = [p.strip() for p in re.split(r"/", s or "") if p.strip()]
    return list(dict.fromkeys(parts))  # stable unique

def _extract_trade_players(trade_list: Iterable[Dict[str, Any]]) -> List[str]:
    pool: Dict[str, None] = {}
    for line in trade_list or []:
        ps = _split_players(str(line.get("players", "")))
        for p in ps:
            pool[p] = None
    return list(pool.keys())

class BaselineGetsFilterMiddleware(BaseHTTPMiddleware):
    """
    If a POST request targets one of the trade-eval endpoints and the body
    includes fragility_mode ~ "trade_delta", injects a note listing the
    traded players so the backend (or the assistant) can limit fragility
    checks to only those players.
    """

    async def dispatch(self, request: Request, call_next):
        try:
            if request.method.upper() == "POST" and request.url.path in _TRADE_PATHS:
                body_bytes = await request.body()
                if body_bytes:
                    data = json.loads(body_bytes.decode("utf-8"))
                else:
                    data = {}

                frag = str(data.get("fragility_mode", "")).lower()
                if "trade_delta" in frag:
                    trade_players = _extract_trade_players(data.get("trade", []))
                    notes = data.get("_notes") or {}
                    notes["fragility_filter"] = "trade_delta"
                    notes["trade_players"] = trade_players
                    data["_notes"] = notes
                    # Optional hint the backend may read:
                    data["fragility_players"] = trade_players
                    # Re-serialize into request body
                    new_bytes = json.dumps(data, separators=(",", ":")).encode("utf-8")
                    async def receive():
                        return {"type": "http.request", "body": new_bytes, "more_body": False}
                    request._receive = receive  # patch the stream for downstream
        except Exception:
            # Silently ignore; never block the request
            pass

        return await call_next(request)
