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

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    base = {"error": "http_error", "status": exc.status_code}
    base["detail"] = exc.detail if not isinstance(exc.detail, str) else exc.detail
    return SafeJSONResponse(status_code=exc.status_code, content=base)

@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    return SafeJSONResponse(
        status_code=500,
        content={"error": "internal_error", "status": 500, "detail": _exc_dict(exc)},
    )


# ---------- Evaluate helpers (rolebook + Wingnut guard) ----------

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

def _wingnut_guard_allows_fe(players: List[str], add_sp: int, leader, accounts: Dict[str, Dict[str, int]]) -> bool:
    """
    Enforce: for any Legends-tagged player where Wingnut84 is Top-3 on the *leaderboard*,
    FE must remain strictly behind Wingnut84 after this GET.
    """
    wingnut_key = "wingnut84"  # case-insensitive
    for p in players:
        pk = _norm(p)
        rows = leader["by_player"].get(pk, [])
        if not rows:
            continue

        # Wingnut SP from leaderboard
        wingnut_sp = 0
        for r in rows:
            if str(r.get("account", "")).lower() == wingnut_key:
                wingnut_sp = int(r.get("sp") or 0)
                break

        if wingnut_sp <= 0:
            continue

        higher = sum(1 for r in rows if int(r.get("sp") or 0) > wingnut_sp)
        wingnut_rank = 1 + higher
        if wingnut_rank > 3:
            continue  # guard only applies when Wingnut is Top-3

        # FE effective SP for this player BEFORE the GET
        fe_lb = _lb_family_sp_for(leader, p, "FinkleIsEinhorn")
        fe_hold = int(accounts.get("FinkleIsEinhorn", {}).get(p, 0))
        fe_eff_before = max(fe_lb, fe_hold)

        # AFTER adding this GET
        fe_eff_after = max(fe_lb, fe_hold + int(add_sp))

        # Must remain strictly below Wingnut's SP
        if fe_eff_after >= wingnut_sp:
            return False
    return True

def _apply_trade_allocation(leader, accounts_before, trade: List, tags: Dict[str, Set[str]]):
    cur = {a: dict(v) for a, v in accounts_before.items()}
    alloc_plan = []

    # GETs: decide destination account
    for line in [l for l in trade if l.side == "GET"]:
        players = split_multi_subject_players(line.players)
        order = _account_order_for_players(players, tags)

        # Determine tag presence
        has_legends = any(_norm(p) in tags.get("LEGENDS", set()) for p in players)
        has_ana = any(_norm(p) in tags.get("ANA", set()) for p in players)
        has_team_dc = any(_norm(p) in tags.get(k, set()) for k in ("DAL", "LAK", "PIT") for p in players)

        # Default account based on tag priority
        to_acct = order[0]

        # Legends: Wingnut84 guard
        guard_info: Dict[str, Any] = {}
        if has_legends and to_acct == "FinkleIsEinhorn":
            allowed = _wingnut_guard_allows_fe(players, int(line.sp), leader, cur)
            guard_info = {"wingnut_guard_applied": True, "allowed": allowed}
            if not allowed:
                # Overflow for Legends when guard blocks FE
                # Prefer DC (consistent with rolebook overflow), else E31
                to_acct = "DusterCrusher"

        # No-tag case: prefer current best holder
        if not (has_legends or has_ana or has_team_dc):
            to_acct = _best_holder_acct(players, leader, cur)

        # Apply the GET
        for p in players:
            cur[to_acct][p] = int(cur[to_acct].get(p, 0)) + int(line.sp)

        meta = {
            "type": "GET",
            "to": to_acct,
            "players": players,
            "sp": int(line.sp),
            "routing_trace": {
                "policy": "tag_first_or_best_holder_with_wingnut_guard",
                "order": order,
            }
        }
        if guard_info:
            meta["routing_trace"]["wingnut_guard"] = guard_info
        alloc_plan.append(meta)

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


# ---------- Probe (diagnostic) route restored ----------

@app.post("/__probe_evaluate")
def probe_family_evaluate(req: Dict[str, Any]):
    """
    Lightweight diagnostic that runs the same inputs pipeline as evaluate(),
    step-by-step, and reports where it fails.
    """
    steps: List[Dict[str, Any]] = []
    prefer = bool(req.get("prefer_env_defaults", True))

    def _add(step: str, **kw):
        d = {"step": step, **kw}
        steps.append(d)
        return d

    # 1) Resolve leaderboard URL
    try:
        url_lb = _pick_url(req.get("leaderboard_url"), "leaderboard", prefer)
        _add("pick_leaderboard_url", ok=True, url=url_lb)
    except Exception as e:
        _add("pick_leaderboard_url", ok=False, error=_exc_dict(e))
        return {"ok": False, "steps": steps}

    # 2) Fetch leaderboard XLSX
    try:
        lb_sheets = fetch_xlsx(url_lb)
        _add("fetch_leaderboard_xlsx", ok=True, sheet_count=len(lb_sheets), sheet_names=list(lb_sheets.keys())[:5])
    except Exception as e:
        _add("fetch_leaderboard_xlsx", ok=False, error=_exc_dict(e))
        return {"ok": False, "steps": steps}

    # 3) Normalize leaderboard
    try:
        leader = normalize_leaderboard(lb_sheets)
        _add("normalize_leaderboard", ok=True, players=len(leader["by_player"]))
    except Exception as e:
        _add("normalize_leaderboard", ok=False, error=_exc_dict(e))
        return {"ok": False, "steps": steps}

    # 4) Load holdings for all family accounts
    try:
        accounts_before = holdings_from_urls(
            req.get("holdings_e31_url"),
            req.get("holdings_dc_url"),
            req.get("holdings_fe_url"),
            prefer,
            req.get("holdings_ud_url"),
        )
        _add("holdings_from_urls", ok=True, counts={k: len(v) for k, v in accounts_before.items()})
    except Exception as e:
        _add("holdings_from_urls", ok=False, error=_exc_dict(e))
        return {"ok": False, "steps": steps}

    # 5) Load tag sheet
    try:
        tags = _load_player_tags(prefer, req.get("player_tags_url"))
        _add("load_player_tags", ok=True, tag_sizes={k: len(v) for k, v in tags.items()})
    except Exception as e:
        _add("load_player_tags", ok=False, error=_exc_dict(e))
        return {"ok": False, "steps": steps}

    # 6) Compute baseline family QP (leaderboard-QP semantics)
    try:
        fam_qp, per_qp, details = compute_family_qp(leader, accounts_before)
        _add("compute_family_qp", ok=True, family_qp_total=int(fam_qp), per_account_qp=per_qp, method=details.get("source"))
    except Exception as e:
        _add("compute_family_qp", ok=False, error=_exc_dict(e))
        return {"ok": False, "steps": steps}

    return {"ok": True, "steps": steps}


# ---------- Evaluate route (with Wingnut guard) ----------

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

    # 2) Allocation (rolebook + Wingnut guard)
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
