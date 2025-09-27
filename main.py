# main.py — loader + robust defaults + strict trade fragility filter
# Version: 4.1.2-reset
from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# ---- import the real app (your domain logic) ----
_core_err: Optional[Exception] = None
try:
    # IMPORTANT: keep your existing app_core.py exactly as-is.
    from app_core import app as core_app
except Exception as e:
    _core_err = e
    core_app = None

if core_app is None:
    raise RuntimeError(f"Could not import app_core.app: {_core_err!r}")

# We will wrap the core app with light defaulting middleware.
app: FastAPI = core_app


# ------------ helpers ------------

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key, default)
    return v if (v is not None and v.strip()) else default

def _resolve_magic_url(token: Optional[str]) -> Optional[str]:
    """
    Convert placeholders the GPT sometimes passes (e.g. 'E31_default')
    into the real URLs from environment. If token is a real URL already,
    return it unchanged.
    """
    if not token or token.startswith("http"):
        return token

    table = {
        "E31_default": _env("DEFAULT_HOLDINGS_E31_URL"),
        "DC_default": _env("DEFAULT_HOLDINGS_DC_URL"),
        "FE_default": _env("DEFAULT_HOLDINGS_FE_URL"),
        "LEADERBOARD_default": _env("DEFAULT_LEADERBOARD_URL"),
        "LEADERBOARD_YDAY_default": _env("DEFAULT_LEADERBOARD_YDAY_URL"),
        "POOL_default": _env("DEFAULT_POOL_COLLECTION_URL"),
        # legacy shorthand that sometimes shows up
        "E31": _env("DEFAULT_HOLDINGS_E31_URL"),
        "DC": _env("DEFAULT_HOLDINGS_DC_URL"),
        "FE": _env("DEFAULT_HOLDINGS_FE_URL"),
    }
    return table.get(token, token)


def _coerce_defaults(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    - Fills URLs from env when placeholders or prefer_env_defaults present
    - Sets default multi_subject_rule and fragility_mode
    - Ensures defend_buffers for family endpoints
    - Adds 'avoid_players_from_trade' to family eval+counter requests so the
      counter review won't re-suggest cards already in trade GETs.
    """
    p = dict(payload or {})

    prefer_env = bool(p.get("prefer_env_defaults", False))

    # URLs present on many endpoints
    for key in [
        "leaderboard_url",
        "leaderboard_today_url",
        "leaderboard_yesterday_url",
        "holdings_url",
        "holdings_e31_url",
        "holdings_dc_url",
        "holdings_fe_url",
        "collection_url",
        "collection_e31_url",
        "collection_dc_url",
        "collection_fe_url",
        "my_collection_url",
    ]:
        v = p.get(key)
        if prefer_env or isinstance(v, str):
            p[key] = _resolve_magic_url(v)

    # If still missing, fill common defaults
    if not p.get("leaderboard_url"):
        p["leaderboard_url"] = _env("DEFAULT_LEADERBOARD_URL")
    if path.endswith("leaderboard_delta_by_urls"):
        if not p.get("leaderboard_today_url"):
            p["leaderboard_today_url"] = _env("DEFAULT_LEADERBOARD_URL")
        if not p.get("leaderboard_yesterday_url"):
            p["leaderboard_yesterday_url"] = _env("DEFAULT_LEADERBOARD_YDAY_URL")

    # default subject rule
    p.setdefault("multi_subject_rule", "full_each_unique")

    # strict trade fragility on trade evaluators unless caller overrides
    if path.endswith("evaluate_by_urls_easystreet31") or path.endswith("family_evaluate_trade_by_urls"):
        p.setdefault("fragility_mode", _env("TRADE_FRAGILITY_DEFAULT", "trade_delta"))

    # defend buffer defaults for family endpoints
    if path.startswith("/family_"):
        try:
            buf = int(_env("DEFAULT_DEFEND_BUFFER_ALL", "15"))
        except Exception:
            buf = 15
        if "defend_buffers" not in p:
            p["defend_buffers"] = {
                "Easystreet31": buf,
                "DusterCrusher": buf,
                "FinkleIsEinhorn": buf,
            }

    # annotate family eval+counter so counter step doesn't re-suggest cards
    if path.endswith("family_eval_and_counter_by_urls"):
        trade = p.get("trade") or []
        get_players = []
        for line in trade:
            if isinstance(line, dict) and line.get("side", "").upper() == "GET":
                name = str(line.get("players", "")).strip()
                if name:
                    get_players.append(name)
        if get_players:
            p.setdefault("avoid_players_from_trade", sorted(set(get_players)))

    return p


class DefaultsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.headers.get("content-type", "").startswith("application/json"):
            raw = await request.body()
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                payload = {}
            patched = _coerce_defaults(request.url.path, payload)

            new_body = json.dumps(patched).encode("utf-8")

            async def receive():
                return {"type": "http.request", "body": new_body, "more_body": False}

            request = Request(request.scope, receive=receive)

        return await call_next(request)


# Install the middleware AFTER we import your core app
app.add_middleware(DefaultsMiddleware)


# Simple health route if your core doesn't provide one
@app.get("/health")
def _health():
    return {"ok": True}


# Expose your effective defaults (nice for smoke tests)
@app.get("/defaults")
def _defaults():
    return {
        "leaderboard_url": _env("DEFAULT_LEADERBOARD_URL"),
        "leaderboard_yesterday_url": _env("DEFAULT_LEADERBOARD_YDAY_URL"),
        "holdings": {
            "E31": _env("DEFAULT_HOLDINGS_E31_URL"),
            "DC": _env("DEFAULT_HOLDINGS_DC_URL"),
            "FE": _env("DEFAULT_HOLDINGS_FE_URL"),
        },
        "collections": {
            "E31": _env("DEFAULT_COLLECTION_E31_URL"),
            "DC": _env("DEFAULT_COLLECTION_DC_URL"),
            "FE": _env("DEFAULT_COLLECTION_FE_URL"),
            "POOL": _env("DEFAULT_POOL_COLLECTION_URL"),
        },
        "target_rivals": [_x for _x in (_env("DEFAULT_TARGET_RIVALS") or "").split(",") if _x],
        "defend_buffer_all": int(_env("DEFAULT_DEFEND_BUFFER_ALL", "15")),
        "trade_fragility_default": _env("TRADE_FRAGILITY_DEFAULT", "trade_delta"),
        "force_family_urls": True,
        "version": "4.1.2-reset",
    }
