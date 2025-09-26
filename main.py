"""
main.py — wrapper that:
  (a) injects default URLs/knobs from environment into request bodies when omitted,
  (b) augments /evaluate_by_urls_easystreet31 responses with BUFFER IMPACT + buffer-aware verdict.

HOW TO DEPLOY
1) Ensure your previous working app is in app_core.py and defines: app = FastAPI(...)
2) Replace your current main.py with this file.
3) Render start command stays: uvicorn main:app --host 0.0.0.0 --port $PORT

This file does NOT change your core business logic; it post-processes responses to value "building buffer".
"""

from __future__ import annotations

import io
import os
import re
import json
from typing import Any, Dict, List, Optional, Callable, Tuple
from collections import defaultdict

import pandas as pd
import requests
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# ------------------------------------------------------------------------------
# 0) Import your original FastAPI app (unchanged business logic)
# ------------------------------------------------------------------------------
_core_import_error = None
core_app = None
try:
    from app_core import app as core_app  # your working app lives here
except Exception as e:
    _core_import_error = e

if core_app is None:
    raise RuntimeError(
        "Could not import your original app from `app_core.py`.\n"
        "Please rename your previous working `main.py` to `app_core.py` at the repo root.\n"
        f"Import error: {repr(_core_import_error)}"
    )

# The FastAPI instance uvicorn will serve.
app: FastAPI = core_app


# ------------------------------------------------------------------------------
# 1) Read environment defaults once (URLs and knobs)
# ------------------------------------------------------------------------------

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if (v is not None and v != "") else default

# URLs (already set up in your service)
DEFAULT_LEADERBOARD_URL       = _env("DEFAULT_LEADERBOARD_URL")
DEFAULT_LEADERBOARD_YDAY_URL  = _env("DEFAULT_LEADERBOARD_YDAY_URL")
DEFAULT_HOLDINGS_E31_URL      = _env("DEFAULT_HOLDINGS_E31_URL")
DEFAULT_HOLDINGS_DC_URL       = _env("DEFAULT_HOLDINGS_DC_URL")
DEFAULT_HOLDINGS_FE_URL       = _env("DEFAULT_HOLDINGS_FE_URL")
DEFAULT_COLLECTION_E31_URL    = _env("DEFAULT_COLLECTION_E31_URL")
DEFAULT_COLLECTION_DC_URL     = _env("DEFAULT_COLLECTION_DC_URL")
DEFAULT_COLLECTION_FE_URL     = _env("DEFAULT_COLLECTION_FE_URL")
DEFAULT_POOL_COLLECTION_URL   = _env("DEFAULT_POOL_COLLECTION_URL")

DEFAULT_TARGET_RIVALS_RAW     = _env("DEFAULT_TARGET_RIVALS", "")
DEFAULT_TARGET_RIVALS: List[str] = (
    [r.strip() for r in DEFAULT_TARGET_RIVALS_RAW.split(",") if r.strip()]
    if DEFAULT_TARGET_RIVALS_RAW is not None else []
)

def _to_int(val: Optional[str], fallback: int) -> int:
    try:
        return int(val) if val is not None else fallback
    except Exception:
        return fallback

DEFAULT_DEFEND_BUFFER_ALL = _to_int(_env("DEFAULT_DEFEND_BUFFER_ALL", "15"), 15)

# Buffer dials (new)
def _to_float(val: Optional[str], fallback: float) -> float:
    try:
        return float(val) if val is not None else fallback
    except Exception:
        return fallback

BUFFER_WEIGHT_PER_SP = _to_float(_env("BUFFER_WEIGHT_PER_SP", "0.15"), 0.15)
BUFFER_CREDIT_CAP_PER_PLAYER = _to_float(_env("BUFFER_CREDIT_CAP_PER_PLAYER", "3"), 3.0)
BUFFER_ACCEPT_THRESHOLD = _to_float(_env("BUFFER_ACCEPT_THRESHOLD", "0.75"), 0.75)
BUFFER_COUNT_UP_TO_DEFEND = bool(str(_env("BUFFER_COUNT_UP_TO_DEFEND", "true")).lower() == "true")


# ------------------------------------------------------------------------------
# 2) Defaults merger (keep your existing behavior: URLs optional in prompts)
# ------------------------------------------------------------------------------

def _merge_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    return None

def _apply_common_defaults(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_url", DEFAULT_LEADERBOARD_URL)
    body.setdefault("holdings_url",    DEFAULT_HOLDINGS_E31_URL)
    body.setdefault("collection_url",  DEFAULT_POOL_COLLECTION_URL)
    body.setdefault("my_collection_url", DEFAULT_COLLECTION_E31_URL)
    if not body.get("target_rivals"):
        if DEFAULT_TARGET_RIVALS:
            body["target_rivals"] = DEFAULT_TARGET_RIVALS
    else:
        maybe = _merge_list(body.get("target_rivals"))
        if maybe is not None:
            body["target_rivals"] = maybe
    body.setdefault("defend_buffer", DEFAULT_DEFEND_BUFFER_ALL)

def _apply_family_defaults(body: Dict[str, Any]) -> None:
    body.setdefault("leaderboard_url", DEFAULT_LEADERBOARD_URL)
    body.setdefault("holdings_e31_url", DEFAULT_HOLDINGS_E31_URL)
    body.setdefault("holdings_dc_url",  DEFAULT_HOLDINGS_DC_URL)
    body.setdefault("holdings_fe_url",  DEFAULT_HOLDINGS_FE_URL)
    body.setdefault("collection_e31_url", DEFAULT_COLLECTION_E31_URL)
    body.setdefault("collection_dc_url",  DEFAULT_COLLECTION_DC_URL)
    body.setdefault("collection_fe_url",  DEFAULT_COLLECTION_FE_URL)
    body.setdefault("defend_buffer_all", DEFAULT_DEFEND_BUFFER_ALL)

# Map routes we auto-fill with defaults
POST_PATHS: Dict[str, Callable[[Dict[str, Any]], None]] = {
    "/evaluate_by_urls_easystreet31":          _apply_common_defaults,
    "/scan_by_urls_easystreet31":              _apply_common_defaults,
    "/scan_rival_by_urls_easystreet31":        _apply_common_defaults,
    "/scan_partner_by_urls_easystreet31":      _apply_common_defaults,
    "/review_collection_by_urls_easystreet31": _apply_common_defaults,
    "/suggest_give_from_collection_by_urls_easystreet31": _apply_common_defaults,
    "/family_evaluate_trade_by_urls":          _apply_family_defaults,
    "/collection_review_family_by_urls":       _apply_family_defaults,
    "/family_transfer_suggestions_by_urls":    _apply_family_defaults,
    "/family_transfer_optimize_by_urls":       _apply_family_defaults,
    "/leaderboard_delta_by_urls":              lambda b: (
        b.setdefault("leaderboard_today_url", DEFAULT_LEADERBOARD_URL),
        b.setdefault("leaderboard_yesterday_url", DEFAULT_LEADERBOARD_YDAY_URL),
        b.setdefault("leaderboard_url", b.get("leaderboard_today_url") or DEFAULT_LEADERBOARD_URL)
    )
}

class MergeDefaultsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method.upper() != "POST":
            return await call_next(request)
        merger = POST_PATHS.get(request.url.path)
        if not merger:
            return await call_next(request)

        try:
            raw = await request.body()
            body = {} if not raw else json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                return await call_next(request)
        except Exception:
            return await call_next(request)

        try:
            merger(body)
        except Exception:
            pass

        # Save merged body for downstream (buffer augmentation will use it)
        request.state.merged_body = body

        new_raw = json.dumps(body).encode("utf-8")

        async def receive():
            return {"type": "http.request", "body": new_raw, "more_body": False}

        request._receive = receive  # type: ignore[attr-defined]
        return await call_next(request)

app.add_middleware(MergeDefaultsMiddleware)


# ------------------------------------------------------------------------------
# 3) Buffer augmentation on /evaluate_by_urls_easystreet31 (post-process response)
# ------------------------------------------------------------------------------

# --- Utilities to load & normalize data ---------------------------------------

def _fetch_xlsx(url: str) -> io.BytesIO:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return io.BytesIO(r.content)

def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()

def _detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c: _norm_name(c) for c in df.columns}
    for c in df.columns:
        low = cols[c]
        for pat in candidates:
            if re.search(pat, low):
                return c
    return None

def _load_holdings_map(url: str) -> Dict[str, float]:
    """
    Returns {player_name_norm: you_sp}
    Tries to find columns like: Player, SP / Subject Points / Total SP
    """
    buf = _fetch_xlsx(url)
    df = pd.read_excel(buf, sheet_name=0, dtype=str)
    # Try to coerce numerics later
    pcol = _detect_col(df, [r"\bplayer\b", r"\bname\b"])
    scol = _detect_col(df, [r"\bsp\b", r"subject\s*_?points", r"total\s*_?sp"])
    if not pcol:
        # Fallback: first column
        pcol = df.columns[0]
    if not scol:
        # Fallback: try any column that looks numeric-heavy
        num_candidates = [c for c in df.columns if c != pcol]
        scol = num_candidates[0] if num_candidates else df.columns[0]

    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        pname = _norm_name(str(row.get(pcol, "")))
        if not pname:
            continue
        try:
            sp = float(str(row.get(scol, "0")).replace(",", "").strip())
        except Exception:
            sp = 0.0
        out[pname] = out.get(pname, 0.0) + sp
    return out

def _load_leaderboard_competitors(url: str, my_user: str = "easystreet31") -> Dict[str, List[float]]:
    """
    Returns {player_name_norm: [competitor_sp1, competitor_sp2, ...]} excluding my_user.
    Accepts either "long" (player,user,sp[,rank]) or "wide top-5" formats.
    """
    buf = _fetch_xlsx(url)
    df = pd.read_excel(buf, sheet_name=0, dtype=str)
    # Try "long" format
    pcol = _detect_col(df, [r"\bplayer\b", r"\bsubject\b"])
    ucol = _detect_col(df, [r"\buser\b", r"\bcollector\b", r"\bowner\b", r"\bname\b"])
    scol = _detect_col(df, [r"\bsp\b", r"subject\s*_?points", r"\bpoints\b", r"total\s*_?sp"])
    if pcol and ucol and scol:
        comp: Dict[str, List[float]] = defaultdict(list)
        for _, row in df.iterrows():
            player = _norm_name(str(row.get(pcol, "")))
            user = _norm_name(str(row.get(ucol, "")))
            try:
                sp = float(str(row.get(scol, "0")).replace(",", "").strip())
            except Exception:
                sp = 0.0
            if player and user and user != _norm_name(my_user):
                comp[player].append(sp)
        # sort desc
        return {k: sorted(v, reverse=True) for k, v in comp.items() if v}

    # Try "wide" top-5 format: Player + pairs of (N User, N SP)
    pcol = _detect_col(df, [r"\bplayer\b", r"\bsubject\b"])
    if not pcol:
        pcol = df.columns[0]
    # Collect all (user,sp) column pairs
    col_lows = {c: _norm_name(c) for c in df.columns}
    rank_pairs: List[Tuple[str, str]] = []
    # Heuristic: find columns that look like "1 user" and "1 sp", ..."5 user","5 sp"
    for n in range(1, 8):  # up to 7 in case
        user_col = None
        sp_col = None
        for c, low in col_lows.items():
            if re.search(rf"\b{n}\b.*(user|name|collector|owner)", low):
                user_col = c
            if re.search(rf"\b{n}\b.*(sp|points)", low):
                sp_col = c
        if user_col and sp_col:
            rank_pairs.append((user_col, sp_col))

    comp: Dict[str, List[float]] = defaultdict(list)
    for _, row in df.iterrows():
        player = _norm_name(str(row.get(pcol, "")))
        if not player:
            continue
        for (uc, sc) in rank_pairs:
            user = _norm_name(str(row.get(uc, "")))
            try:
                sp = float(str(row.get(sc, "0")).replace(",", "").strip())
            except Exception:
                sp = 0.0
            if user and user != _norm_name(my_user) and sp > 0:
                comp[player].append(sp)
    return {k: sorted(v, reverse=True) for k, v in comp.items() if v}

def _parse_trade_deltas(trade_lines: List[Dict[str, Any]], multi_rule: str = "full_each_unique") -> Dict[str, float]:
    """
    Returns net delta per player for your account: +sp for GET, -sp for GIVE.
    multi_rule 'full_each_unique' => full SP to every distinct player on the card.
    """
    deltas: Dict[str, float] = defaultdict(float)
    for line in trade_lines or []:
        side = str(line.get("side", "")).strip().upper()
        sp = float(line.get("sp", 0) or 0)
        players_field = str(line.get("players", "") or "")
        # Split on '/', '&', ',', ' and '
        raw_names = re.split(r"\s*/\s*|\s*&\s*|,\s*|\s+and\s+", players_field.strip())
        names = sorted(set(_norm_name(n) for n in raw_names if n))
        if not names or sp <= 0:
            continue
        for nm in names:
            if side == "GET":
                deltas[nm] += sp
            elif side == "GIVE":
                deltas[nm] -= sp
    return deltas

def _rank_and_qp(you_sp: float, comp_sps: List[float]) -> Tuple[int, int]:
    """Conservative: any tie with a competitor counts as behind (you need +1)."""
    greater = sum(1 for v in comp_sps if v > you_sp)
    equal   = sum(1 for v in comp_sps if v == you_sp)
    rank = 1 + greater + (1 if equal > 0 else 0)
    qp = 5 if rank == 1 else (3 if rank == 2 else (1 if rank == 3 else 0))
    return rank, qp

def _buffer_credit(margin_before: float, margin_after: float, defend: int) -> float:
    """
    Compute buffer credit from margin delta, only counting up to defend if enabled.
    """
    mb = margin_before
    ma = margin_after
    if BUFFER_COUNT_UP_TO_DEFEND:
        mb = min(mb, defend)
        ma = min(ma, defend)
    gain_sp = ma - mb  # can be negative
    credit = gain_sp * BUFFER_WEIGHT_PER_SP
    # Positive credit capped per player:
    if credit > 0:
        credit = min(credit, BUFFER_CREDIT_CAP_PER_PLAYER)
    return credit

# --- Middleware that augments the evaluator response --------------------------

class BufferAugmentMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Only target the single-account evaluator
        if request.method.upper() != "POST" or request.url.path != "/evaluate_by_urls_easystreet31":
            return await call_next(request)

        # Grab the merged body from the defaults middleware
        body = getattr(request.state, "merged_body", None)
        if not isinstance(body, dict):
            return await call_next(request)

        # Call downstream first to get the original evaluation JSON
        response = await call_next(request)

        # We only handle JSON responses
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" not in content_type:
            return response

        # Read response body (consume iterator, then rebuild Response)
        resp_bytes = b""
        async for chunk in response.body_iterator:
            resp_bytes += chunk

        try:
            base = json.loads(resp_bytes.decode("utf-8"))
        except Exception:
            # If we can't parse, just return original
            return Response(content=resp_bytes, status_code=response.status_code,
                            headers=dict(response.headers), media_type="application/json")

        # ----------------- Compute BUFFER IMPACT locally -----------------
        try:
            leaderboard_url = body.get("leaderboard_url") or DEFAULT_LEADERBOARD_URL
            holdings_url    = body.get("holdings_url") or DEFAULT_HOLDINGS_E31_URL
            defend_buffer   = int(body.get("defend_buffer") or DEFAULT_DEFEND_BUFFER_ALL)
            trade_lines     = body.get("trade") or []
            multi_rule      = body.get("multi_subject_rule") or "full_each_unique"

            # Load data
            you_map = _load_holdings_map(holdings_url)                    # {player: you_sp}
            comp_map = _load_leaderboard_competitors(leaderboard_url)     # {player: [competitor sps...]}

            # Which players are impacted by the trade
            deltas = _parse_trade_deltas(trade_lines, multi_rule)
            impacted = sorted(deltas.keys())

            buffer_details: List[Dict[str, Any]] = []
            total_buffer_credit = 0.0
            total_buffer_penalty = 0.0
            delta_qp_total = 0

            for pname in impacted:
                you_before = float(you_map.get(pname, 0.0))
                you_after  = you_before + float(deltas.get(pname, 0.0))
                comps = comp_map.get(pname, [])
                top_comp = comps[0] if comps else 0.0

                # Rank/QP before/after (for final ΔQP sanity per impacted player)
                r_b, qp_b = _rank_and_qp(you_before, comps)
                r_a, qp_a = _rank_and_qp(you_after,  comps)
                d_qp = qp_a - qp_b
                delta_qp_total += d_qp

                # Buffer applies only if you were actually 1st strictly before
                if you_before > top_comp:
                    margin_before = you_before - top_comp
                    margin_after  = max(0.0, you_after - top_comp)
                    credit = _buffer_credit(margin_before, margin_after, defend_buffer)
                    if credit >= 0:
                        total_buffer_credit += credit
                    else:
                        total_buffer_penalty += credit  # negative
                    buffer_details.append({
                        "player": pname,
                        "you_sp_before": you_before,
                        "you_sp_after": you_after,
                        "second_sp": top_comp,
                        "margin_before": margin_before,
                        "margin_after": margin_after,
                        "buffer_delta_sp": (margin_after - margin_before) if BUFFER_COUNT_UP_TO_DEFEND else (you_after - you_before),
                        "credit": round(credit, 3),
                        "rank_before": r_b,
                        "rank_after": r_a,
                        "qp_before": qp_b,
                        "qp_after": qp_a,
                        "delta_qp": d_qp
                    })
                else:
                    # Not 1st before → no buffer credit (but show rank/qp context)
                    buffer_details.append({
                        "player": pname,
                        "you_sp_before": you_before,
                        "you_sp_after": you_after,
                        "second_sp": top_comp,
                        "margin_before": you_before - top_comp,
                        "margin_after": you_after - top_comp,
                        "buffer_delta_sp": 0,
                        "credit": 0.0,
                        "rank_before": r_b,
                        "rank_after": r_a,
                        "qp_before": qp_b,
                        "qp_after": qp_a,
                        "delta_qp": d_qp
                    })

            buffer_net = total_buffer_credit + total_buffer_penalty  # penalty is ≤ 0
            # Prefer ΔQP from base response if present; else fall back to our impacted-only ΔQP
            base_qp_delta = None
            try:
                # Try common locations/keys used in your responses
                if isinstance(base.get("qp_summary"), dict):
                    base_qp_delta = base["qp_summary"].get("net_delta_qp")
                if base_qp_delta is None and isinstance(base.get("summary"), dict):
                    base_qp_delta = base["summary"].get("net_delta_qp")
            except Exception:
                pass
            delta_qp_effective = base_qp_delta if isinstance(base_qp_delta, (int, float)) else delta_qp_total

            composite = float(delta_qp_effective) + float(buffer_net)

            # Build buffer block
            buffer_block = {
                "config": {
                    "defend_buffer": defend_buffer,
                    "weight_per_sp": BUFFER_WEIGHT_PER_SP,
                    "cap_per_player": BUFFER_CREDIT_CAP_PER_PLAYER,
                    "count_up_to_defend": BUFFER_COUNT_UP_TO_DEFEND,
                    "accept_threshold_if_qp_zero": BUFFER_ACCEPT_THRESHOLD,
                },
                "details": buffer_details,
                "totals": {
                    "delta_qp_on_impacted_players": delta_qp_total,
                    "buffer_credit_sum": round(total_buffer_credit, 3),
                    "buffer_penalty_sum": round(total_buffer_penalty, 3),
                    "buffer_net": round(buffer_net, 3),
                    "composite_score": round(composite, 3),
                }
            }

            # Attach to response
            base["buffer_impact"] = buffer_block

            # Verdict logic: if base verdict neutral/decline due to QP = 0 but composite strong, flip to accept
            verdict_text = str(base.get("verdict") or "").strip().lower()
            # Pull a numeric ΔQP if base exposed it, else our impacted-only ΔQP
            dq = float(delta_qp_effective or 0.0)

            if dq >= 1:
                # Already a QP win; just annotate that buffer supports the win
                base["verdict_note"] = f"QP-positive; buffer support {round(buffer_net,3)}."
            elif dq <= -1:
                # QP-loser stays a reject, but show buffer context
                base["verdict_note"] = f"QP-negative; buffer impact {round(buffer_net,3)}."
            else:
                # ΔQP == 0 → use composite
                if composite >= BUFFER_ACCEPT_THRESHOLD:
                    base["verdict"] = "ACCEPT (buffer-positive)"
                    base["verdict_note"] = f"Composite {round(composite,3)} ≥ {BUFFER_ACCEPT_THRESHOLD}; buffer nets {round(buffer_net,3)}."
                else:
                    # keep original; annotate
                    base["verdict_note"] = f"ΔQP=0; composite {round(composite,3)} (< {BUFFER_ACCEPT_THRESHOLD})."

            new_bytes = json.dumps(base, ensure_ascii=False).encode("utf-8")
            return Response(content=new_bytes, status_code=response.status_code,
                            headers={**dict(response.headers), "content-type": "application/json"})
        except Exception:
            # If anything goes wrong, return original response unchanged
            return Response(content=resp_bytes, status_code=response.status_code,
                            headers=dict(response.headers), media_type="application/json")

# Attach the buffer augmentation middleware
app.add_middleware(BufferAugmentMiddleware)


# ------------------------------------------------------------------------------
# 4) Diagnostics
# ------------------------------------------------------------------------------

@app.get("/defaults")
def defaults_echo() -> Dict[str, Any]:
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
        "buffer": {
            "weight_per_sp": BUFFER_WEIGHT_PER_SP,
            "cap_per_player": BUFFER_CREDIT_CAP_PER_PLAYER,
            "count_up_to_defend": BUFFER_COUNT_UP_TO_DEFEND,
            "accept_threshold_if_qp_zero": BUFFER_ACCEPT_THRESHOLD,
        },
        "note": "Defaults apply only when a field is omitted in the POST body."
    }
