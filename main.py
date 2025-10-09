import json
import math
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

# Import the FastAPI app and helpers from your service core
from app_core import (
    app as core_app,
    _pick_url, fetch_xlsx, normalize_leaderboard,
    holdings_from_urls, _load_player_tags, compute_family_qp,
    _rank_and_buffer_full_leader, _lb_family_sp_for, _rank_context_smallset,
    split_multi_subject_players, FAMILY_ACCOUNTS,
    FamilyEvaluateTradeReq,
)

# ---------- Strict, RFC-compliant JSON for every route ----------

def _sanitize(obj: Any) -> Any:
    """Convert non-finite numbers to None; normalize numpy/pandas scalars."""
    try:
        import numpy as np
        NP_FLOAT = (np.floating,)
        NP_INT = (np.integer,)
    except Exception:
        NP_FLOAT = tuple()
        NP_INT = tuple()

    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if NP_FLOAT and isinstance(obj, NP_FLOAT):
        v = float(obj)
        return v if math.isfinite(v) else None

    if NP_INT and isinstance(obj, NP_INT):
        return int(obj)

    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(v) for v in obj]

    try:
        import pandas as pd
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


class SafeJSONResponse(JSONResponse):
    """Response class that emits strict RFC-8259 JSON (no NaN/Inf)."""
    media_type = "application/json"
    def render(self, content: Any) -> bytes:
        return json.dumps(_sanitize(content), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _exc_dict(exc: Exception) -> Dict[str, Any]:
    t = type(exc).__name__
    msg = str(exc)
    return {"type": t, "message": msg}


# ---------- Adopt core app & set global behavior ----------

app = core_app

# CORS for swagger/curl/GPT/etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Force SafeJSONResponse on ALL currently-registered API routes
for route in list(app.routes):
    if isinstance(route, APIRoute):
        route.response_class = SafeJSONResponse


# ---------- JSON error handlers (jq-parsable failures) ----------

@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    # Avoid HTML – return structured JSON with error type/message
    return SafeJSONResponse(
        status_code=500,
        content={"error": "internal_error", "status": 500, "detail": _exc_dict(exc)},
    )


# ---------- Replace /family_evaluate_trade_by_urls with a safe implementation ----------

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _account_order_for_players(players: List[str], tags: Dict[str, Set[str]]) -> List[str]:
    """Rolebook order: Legends->FE, ANA->UD, DAL/LAK/PIT->DC, else E31 fallback."""
    pl = [_norm(p) for p in players]
    any_legends = any(_norm(p) in tags.get("LEGENDS", set()) for p in pl)
    any_ana = any(_norm(p) in tags.get("ANA", set()) for p in pl)
    any_team_dc = any(_norm(p) in tags.get(k, set()) for k in ("DAL", "LAK", "PIT") for p in pl)
    if any_legends:
        return ["FinkleIsEinhorn", "DusterCrusher", "Easystreet31", "UpperDuck"]
    if any_ana:
        return ["UpperDuck", "Easystreet31", "DusterCrusher", "FinkleIsEinhorn"]
    if any_team_dc:
        return ["DusterCrusher", "Easystreet31", "FinkleIsEinhorn", "UpperDuck"]
    return ["Easystreet31", "DusterCrusher", "FinkleIsEinhorn", "UpperDuck"]

def _best_holder_acct(players: List[str], leader, accounts: Dict[str, Dict[str, int]]) -> str:
    totals = {a: 0 for a in FAMILY_ACCOUNTS}
    for p in players:
        for a in FAMILY_ACCOUNTS:
            lb = _lb_family_sp_for(leader, p, a)
            hold = int(accounts.get(a, {}).get(p, 0))
            totals[a] += max(lb, hold)
    # Stable tiebreaker by fixed order
    order = ["Easystreet31", "DusterCrusher", "FinkleIsEinhorn", "UpperDuck"]
    return max(order, key=lambda a: (totals[a], -order.index(a)))

def _apply_trade_allocation(leader, accounts_before, trade: List, tags: Dict[str, Set[str]]):
    cur = {a: dict(v) for a, v in accounts_before.items()}
    alloc_plan = []

    # GETs: decide destination account
    for line in [l for l in trade if l.side == "GET"]:
        players = split_multi_subject_players(line.players)
        order = _account_order_for_players(players, tags)
        # tags-first (Wingnut trailing guard can be added later; minimal safe version here)
        to_acct = order[0]

        # For no-tag case, prefer current best holder
        if not any((_norm(p) in tags.get("LEGENDS", set())
                    or _norm(p) in tags.get("ANA", set())
                    or _norm(p) in tags.get("DAL", set())
                    or _norm(p) in tags.get("LAK", set())
                    or _norm(p) in tags.get("PIT", set())) for p in players):
            to_acct = _best_holder_acct(players, leader, cur)

        for p in players:
            cur[to_acct][p] = int(cur[to_acct].get(p, 0)) + int(line.sp)

        alloc_plan.append({
            "type": "GET",
            "to": to_acct,
            "players": players,
            "sp": int(line.sp),
            "routing_trace": {"policy": "tag_first_or_best_holder", "order": order}
        })

    # GIVEs: subtract from trade_account (clamped at 0)
    for line in [l for l in trade if l.side == "GIVE"]:
        for p in split_multi_subject_players(line.players):
            cur_amount = int(cur.get(trade.trade_account, {}).get(p, 0))
            cur[trade.trade_account][p] = max(0, cur_amount - int(line.sp))

    return cur, alloc_plan

def _effective_maps_for_player(leader, p: str, accounts_before, cur):
    lb_base = {a: _lb_family_sp_for(leader, p, a) for a in FAMILY_ACCOUNTS}
    hold_b = {a: int(accounts_before.get(a, {}).get(p, 0)) for a in FAMILY_ACCOUNTS}
    eff_b  = {a: max(lb_base[a], hold_b[a]) for a in FAMILY_ACCOUNTS}
    eff_a  = {a: int(cur.get(a, {}).get(p, 0)) for a in FAMILY_ACCOUNTS}
    return eff_b, eff_a

# Remove existing route if present (avoid double handlers)
for i, r in list(enumerate(app.router.routes)):
    if isinstance(r, APIRoute) and "/family_evaluate_trade_by_urls" in {getattr(r, "path", None)}:
        del app.router.routes[i]

@app.post("/family_evaluate_trade_by_urls")
def family_evaluate_trade_by_urls_safe(req: FamilyEvaluateTradeReq):
    # 1) Inputs
    tags_map = _load_player_tags(req.prefer_env_defaults, req.player_tags_url)
    leader = normalize_leaderboard(fetch_xlsx(_pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)))
    accounts_before = holdings_from_urls(
        req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url,
        req.prefer_env_defaults, req.holdings_ud_url
    )
    fam0, per0, det0 = compute_family_qp(leader, accounts_before)

    give_requested: Dict[str, int] = defaultdict(int)
    for line in [l for l in req.trade if l.side == "GIVE"]:
        for p in split_multi_subject_players(line.players):
            give_requested[p] += int(line.sp)

    # 2) Allocation (rolebook)
    cur, alloc_plan = _apply_trade_allocation(leader, accounts_before, req.trade, tags_map)

    # 3) Apply GIVEs to trade_account (ensure clamped non-negative)
    for line in [l for l in req.trade if l.side == "GIVE"]:
        for p in split_multi_subject_players(line.players):
            have = int(cur.get(req.trade_account, {}).get(p, 0))
            cur[req.trade_account][p] = max(0, have - int(line.sp))

    fam1, per1, det1 = compute_family_qp(leader, cur)

    touched: Set[str] = set()
    for tl in req.trade:
        for pl in split_multi_subject_players(tl.players):
            touched.add(pl)

    # Ownership warnings
    ownership_warnings = []
    for p, want in sorted(give_requested.items(), key=lambda kv: kv[0].lower()):
        have = int(accounts_before.get(req.trade_account, {}).get(p, 0))
        if want > have:
            ownership_warnings.append({
                "account": req.trade_account, "player": p,
                "attempted_give_sp": int(want), "owned_sp_before": int(have),
                "note": "GIVE exceeds owned SP; family total may not drop as expected."
            })

    # Player changes
    rows = []
    tot_buf = 0
    for pl in sorted(touched):
        eff_b, eff_a = _effective_maps_for_player(leader, pl, accounts_before, cur)
        sp_b = int(sum(eff_b.values()))
        sp_a = int(sum(eff_a.values()))
        d_sp = sp_a - sp_b

        eff_before_map = {a: {pl: eff_b[a]} for a in FAMILY_ACCOUNTS}
        eff_after_map  = {a: {pl: eff_a[a]} for a in FAMILY_ACCOUNTS}
        ctx_b = _rank_context_smallset(pl, leader, eff_before_map)
        ctx_a = _rank_context_smallset(pl, leader, eff_after_map)
        d_qp = int(ctx_a["family_qp_player"] - ctx_b["family_qp_player"])

        r_b_full, buf_b_full, _, _ = _rank_and_buffer_full_leader(pl, leader, {a: eff_b[a] for a in FAMILY_ACCOUNTS})
        r_a_full, buf_a_full, _, _ = _rank_and_buffer_full_leader(pl, leader, {a: eff_a[a] for a in FAMILY_ACCOUNTS})
        d_buf = None if (buf_b_full is None and buf_a_full is None) else int((buf_a_full or 0) - (buf_b_full or 0))
        if d_buf is not None:
            tot_buf += d_buf

        rows.append({
            "player": pl,
            "sp_before": sp_b,
            "sp_after":  sp_a,
            "delta_sp":  d_sp,
            "per_account_sp_before": eff_b,
            "per_account_sp_after":  eff_a,
            "per_account_sp_delta":  {a: eff_a[a] - eff_b[a] for a in FAMILY_ACCOUNTS},
            "best_rank_before": r_b_full,
            "best_rank_after":  r_a_full,
            "best_rank_before_label": "—" if (r_b_full is None) else str(r_b_full),
            "best_rank_after_label":  "—" if (r_a_full is None) else str(r_a_full),
            "buffer_before": None if buf_b_full is None else int(buf_b_full),
            "buffer_after":  None if buf_a_full is None else int(buf_a_full),
            "delta_buffer":  None if d_buf is None else int(d_buf),
            "qp_before": int(ctx_b["family_qp_player"]),
            "qp_after":  int(ctx_a["family_qp_player"]),
            "delta_qp":  d_qp
        })

    rows.sort(key=lambda r: (-r["delta_qp"], -r["delta_sp"], r["player"].lower()))
    totals = {
        "delta_sp": int(sum(r["delta_sp"] for r in rows)),
        "delta_buffer": int(tot_buf),
        "delta_qp": int(fam1 - fam0)
    }

    verdict = "ACCEPT" if (fam1 - fam0) > 0 else ("CAUTION" if (fam1 - fam0) == 0 else "DECLINE")

    return {
        "ok": True,
        "allocation_plan": alloc_plan,
        "player_changes": rows,
        "total_changes": totals,
        "verdict": verdict,
        "ownership_warnings": ownership_warnings,
        "family_qp_before": int(fam0),
        "family_qp_after": int(fam1),
    }


# ---------- Small probe ----------

@app.get("/info")
def info():
    return {
        "ok": True,
        "title": getattr(app, "title", "Ultimate Quest Service (Small-Payload API)"),
        "version": getattr(app, "version", "unknown"),
        "default_response_class": "SafeJSONResponse",
    }
