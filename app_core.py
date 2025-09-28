# app_core.py — Ultimate Quest Service (Small-Payload API)
# Version: 4.3.0  (adds /family_internal_transfers_optimize_by_urls; keeps prior fixes)
#
# Endpoints:
#   • GET  /defaults
#   • POST /family_evaluate_trade_by_urls
#   • POST /family_trade_plus_counter_by_urls
#   • POST /leaderboard_delta_by_urls
#   • POST /leaderboard_delta                 (convenience; uses env defaults)
#   • POST /family_internal_transfers_optimize_by_urls   <-- NEW (greedy optimizer for internal transfers)
#
# Notes:
#   • Accepts Google Sheets "edit/view" links; converts to export XLSX internally.
#   • Trades: fragility is TRADE-ONLY; we report ONLY fragility CREATED by the trade (plus a friendly note).
#   • Leaderboard delta: robust case-insensitive user matching with punctuation/parentheses stripped.

from __future__ import annotations

import io
import os
import re
import math
from typing import Dict, Any, Optional, List, Tuple, Literal, Set
from collections import defaultdict, Counter

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_VERSION = "4.3.0"
APP_TITLE = "Ultimate Quest Service (Small-Payload API)"

FAMILY_ACCOUNTS = ["Easystreet31", "DusterCrusher", "FinkleIsEinhorn"]

# ---------- Defaults (from env) ----------
DEFAULT_LINKS = {
    "leaderboard": os.getenv("DEFAULT_LEADERBOARD_URL", ""),
    "leaderboard_yday": os.getenv("DEFAULT_LEADERBOARD_YDAY_URL", ""),
    "holdings_e31": os.getenv("DEFAULT_HOLDINGS_E31_URL", ""),
    "holdings_dc":  os.getenv("DEFAULT_HOLDINGS_DC_URL", ""),
    "holdings_fe":  os.getenv("DEFAULT_HOLDINGS_FE_URL", ""),
    "collection_e31": os.getenv("DEFAULT_COLLECTION_E31_URL", ""),
    "collection_dc":  os.getenv("DEFAULT_COLLECTION_DC_URL", ""),
    "collection_fe":  os.getenv("DEFAULT_COLLECTION_FE_URL", ""),
    "pool_collection": os.getenv("DEFAULT_POOL_COLLECTION_URL", ""),
}
DEFAULT_RIVALS = [u.strip() for u in (os.getenv("DEFAULT_TARGET_RIVALS", "") or
                                      "chfkyle,Tfunite,FireRanger,VJV5,Erikk,tommyknockrs76").split(",") if u.strip()]
DEFAULT_DEFEND_BUFFER = int(os.getenv("DEFAULT_DEFEND_BUFFER_ALL", "15"))

# ---------- App ----------
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# ---------- Helpers ----------
def _pick_url(explicit: Optional[str], key: str, prefer_env_defaults: bool) -> str:
    url = (explicit or "").strip()
    if not url:
        url = DEFAULT_LINKS[key]
    return _sanitize_gsheet_url(url)

def _sanitize_gsheet_url(url: str) -> str:
    if not url:
        return url
    try:
        if "docs.google.com/spreadsheets/d/" in url:
            m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
            if m:
                doc_id = m.group(1)
                gid_match = re.search(r"[?&#]gid=(\d+)", url)
                gid_part = f"&gid={gid_match.group(1)}" if gid_match else ""
                return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=xlsx{gid_part}"
    except Exception:
        pass
    return url

def _as_int(x: Any) -> int:
    try:
        if pd.isna(x) or x is None or x == "": return 0
        if isinstance(x, (int, np.integer)): return int(x)
        if isinstance(x, float) and math.isfinite(x): return int(round(x))
        s = str(x).replace(",", "").strip()
        if s == "": return 0
        return int(round(float(s)))
    except Exception:
        return 0

def fetch_xlsx(url: str) -> Dict[str, pd.DataFrame]:
    if not url:
        raise HTTPException(status_code=400, detail="Missing XLSX URL.")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = io.BytesIO(resp.content)
        xls = pd.ExcelFile(data)
        out = {}
        for name in xls.sheet_names:
            df = xls.parse(name)
            df.columns = [str(c).strip() for c in df.columns]
            out[name] = df
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load XLSX from URL: {url} ({e})")

def _norm_player(p: Any) -> str:
    if pd.isna(p) or p is None: return ""
    return re.sub(r"\s+", " ", str(p).strip())

def _canon_user(u: Any) -> str:
    if pd.isna(u) or u is None: return ""
    return re.sub(r"\s+", "", str(u).strip())

def _canon_user_strong(u: Any) -> str:
    """ Strong canonicalization for cross-day matching and rival matching. """
    if pd.isna(u) or u is None: return ""
    s = str(u)
    s = re.sub(r"\(.*?\)", "", s)         # remove parentheses
    s = re.sub(r"\s+", "", s)             # remove whitespace
    s = re.sub(r"[^A-Za-z0-9]+", "", s)   # keep alphanumerics only
    return s.lower()

def _unique_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out

def split_multi_subject_players(players_text: str) -> List[str]:
    if not players_text:
        return []
    parts = re.split(r"\s*(?:/|&|\+|,| and |\||—|–)\s*", str(players_text))
    parts = [_norm_player(p) for p in parts if _norm_player(p)]
    return _unique_preserve(parts)

def normalize_leaderboard(sheets: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert a leaderboard workbook into:
      { "<player>": [ {"user": "<user col or header>", "sp": N}, ... ] }
    Accepts either long form (player/user/sp) or wide form (player + many user columns).
    """
    candidate = None
    for name, df in sheets.items():
        lower = [str(c).lower() for c in df.columns]
        if any("player" in c for c in lower):
            candidate = df
            break
    if candidate is None:
        candidate = list(sheets.values())[0]
    df = candidate.copy()
    df.columns = [str(c).strip() for c in df.columns]

    player_col = next((c for c in df.columns if "player" in c.lower()), None)
    user_col   = next((c for c in df.columns if any(k in c.lower() for k in ["user","owner","name"])), None)
    sp_col     = next((c for c in df.columns if any(k in c.lower() for k in ["sp","points"])), None)

    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if player_col and user_col and sp_col:
        # Long form
        for _, row in df.iterrows():
            p = _norm_player(row.get(player_col, ""))
            u = str(row.get(user_col, "")).strip()
            sp = _as_int(row.get(sp_col, 0))
            if p and u != "":
                out[p].append({"user": u, "sp": sp})
    else:
        # Wide form
        if not player_col:
            player_col = df.columns[0]
        user_cols = [c for c in df.columns if c != player_col]
        for _, row in df.iterrows():
            p = _norm_player(row.get(player_col, ""))
            if not p:
                continue
            for uc in user_cols:
                u = uc
                sp = _as_int(row.get(uc, 0))
                out[p].append({"user": u, "sp": sp})

    # Deduplicate to best SP per user and sort
    for p, lst in out.items():
        best: Dict[str, int] = {}
        for e in lst:
            u = str(e["user"])
            if u not in best or e["sp"] > best[u]:
                best[u] = _as_int(e["sp"])
        merged = [{"user": u, "sp": best[u]} for u in best]
        merged.sort(key=lambda x: (-x["sp"], x["user"].lower()))
        out[p] = merged
    return out

def parse_holdings(sheets: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """
    Return { player -> SP } for one account holdings workbook.
    """
    df = None
    for name, d in sheets.items():
        if any("player" in c.lower() or "subject" in c.lower() for c in d.columns):
            df = d
            break
    if df is None:
        df = list(sheets.values())[0]
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    player_col = next((c for c in df.columns if any(k in c.lower() for k in ["player","subject"])), df.columns[0])
    sp_col = next((c for c in df.columns if any(k in c.lower() for k in ["sp","points"])), None)
    if sp_col is None:
        for c in reversed(df.columns):
            if df[c].dtype != object:
                sp_col = c
                break
        sp_col = sp_col or df.columns[-1]
    agg: Dict[str, int] = {}
    for _, row in df.iterrows():
        p = _norm_player(row.get(player_col, ""))
        sp = _as_int(row.get(sp_col, 0))
        if p and sp:
            agg[p] = agg.get(p, 0) + sp
    return agg

def parse_collection(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Normalize collection workbook to columns: card, no, players, sp, qty
    """
    df = list(sheets.values())[0].copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_card = next((c for c in df.columns if any(k in c.lower() for k in ["card","title","item"])), df.columns[0])
    col_no   = next((c for c in df.columns if c.lower() in ("no","#","num","id","index")), None)
    col_pl   = next((c for c in df.columns if any(k in c.lower() for k in ["players","player","subject"])), None)
    col_sp   = next((c for c in df.columns if any(k in c.lower() for k in ["sp","points"])), None)
    col_qty  = next((c for c in df.columns if any(k in c.lower() for k in ["qty","quantity","count"])), None)
    out = pd.DataFrame({
        "card": df.get(col_card, ""),
        "no": df.get(col_no, ""),
        "players": df.get(col_pl, ""),
        "sp": df.get(col_sp, 0).map(_as_int),
        "qty": df.get(col_qty, 1).map(_as_int),
    })
    out["players"] = out["players"].fillna("").astype(str)
    out["card"] = out["card"].fillna("").astype(str)
    out["no"] = out["no"].fillna("").astype(str)
    out["sp"] = out["sp"].fillna(0).astype(int)
    out["qty"] = out["qty"].fillna(1).astype(int)
    out = out[out["card"].str.strip() != ""].copy()
    out.reset_index(drop=True, inplace=True)
    return out

# ---------- QP math ----------
def qp_for_rank(rank: int) -> int:
    return 5 if rank == 1 else 3 if rank == 2 else 1 if rank == 3 else 0

def compute_family_qp(leader: Dict[str, List[Dict[str, Any]]],
                      accounts_sp: Dict[str, Dict[str, int]]
) -> Tuple[int, Dict[str, int], Dict[str, Dict[str, Any]]]:
    """
    Return:
      family_qp_total,
      per_account_qp: {acct -> QP},
      per_account_details: {acct -> { player -> {sp, rank, gap_to_first, margin_if_first} } }
    """
    per_account_qp: Dict[str, int] = {a: 0 for a in FAMILY_ACCOUNTS}
    per_account_details: Dict[str, Dict[str, Any]] = {a: {} for a in FAMILY_ACCOUNTS}

    all_players = set(leader.keys())
    for a in FAMILY_ACCOUNTS:
        all_players.update(accounts_sp.get(a, {}).keys())

    for player in all_players:
        lst = leader.get(player, [])
        top_users = [e["user"] for e in lst[:3]]
        top_sp    = [e["sp"]  for e in lst[:3]]
        first_sp = top_sp[0] if top_sp else 0

        fam_sp = {a: accounts_sp.get(a, {}).get(player, 0) for a in FAMILY_ACCOUNTS}

        merged = [{"user": a, "sp": fam_sp[a]} for a in FAMILY_ACCOUNTS if fam_sp[a] > 0]
        merged += [{"user": u, "sp": s} for u, s in zip(top_users, top_sp)]
        merged.sort(key=lambda x: (-x["sp"], str(x["user"]).lower()))

        best: Dict[str, int] = {}
        for e in merged:
            u = str(e["user"])
            if u not in best:
                best[u] = _as_int(e["sp"])

        ordered = sorted(best.items(), key=lambda x: (-x[1], x[0].lower()))
        ranks_for_user: Dict[str, int] = {}
        last_sp = None
        current_rank = 0
        seen = 0
        for u, sp in ordered:
            seen += 1
            if last_sp is None or sp < last_sp:
                current_rank = seen
            ranks_for_user[u] = current_rank
            last_sp = sp

        for a in FAMILY_ACCOUNTS:
            r = ranks_for_user.get(a, 9999)
            sp = fam_sp[a]
            qp = qp_for_rank(r)
            per_account_qp[a] += qp
            gap_to_first = max(first_sp - sp, 0) if first_sp else 0
            margin_if_first = (sp - (top_sp[1] if len(top_sp) > 1 else 0)) if r == 1 else None
            per_account_details[a][player] = {
                "sp": sp, "rank": r, "gap_to_first": gap_to_first, "margin_if_first": margin_if_first
            }

    family_qp_total = sum(per_account_qp.values())
    return family_qp_total, per_account_qp, per_account_details

def apply_card_sp_to_account(acct_sp: Dict[str, int], players: List[str], sp: int, add: bool) -> None:
    unique_players = []
    seen = set()
    for p in players:
        if p.lower() not in seen:
            seen.add(p.lower())
            unique_players.append(p)
    for p in unique_players:
        acct_sp[p] = acct_sp.get(p, 0) + (sp if add else -sp)
        if acct_sp[p] <= 0:
            acct_sp.pop(p, None)

def clone_accounts_sp(accounts_sp: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    return {a: dict(v) for a, v in accounts_sp.items()}

# --- Fragility helpers (trade-only) ---
def _created_fragile_firsts(
    det_before: Dict[str, Dict[str, Any]],
    det_after: Dict[str, Dict[str, Any]],
    buffers: Dict[str, int],
    restrict_players: Optional[Set[str]] = None
) -> List[str]:
    alerts: List[str] = []
    for acct in FAMILY_ACCOUNTS:
        buf = buffers.get(acct, DEFAULT_DEFEND_BUFFER)
        after_map = det_after.get(acct, {}) or {}
        before_map = det_before.get(acct, {}) or {}
        for player, ainfo in after_map.items():
            if restrict_players and player not in restrict_players:
                continue
            a_rank = ainfo.get("rank", 9999)
            a_margin = ainfo.get("margin_if_first")
            frag_after = (a_rank == 1 and a_margin is not None and a_margin < buf)
            if not frag_after:
                continue
            binfo = before_map.get(player, {})
            b_rank = binfo.get("rank", 9999)
            b_margin = binfo.get("margin_if_first")
            frag_before = (b_rank == 1 and b_margin is not None and b_margin < buf)
            if frag_after and not frag_before:
                alerts.append(f"{acct}: {player} (margin {int(a_margin)})")
    return sorted(alerts)

# ---------- Pydantic models ----------
class TradeLine(BaseModel):
    side: Literal["GET","GIVE"]
    players: str
    sp: int

class FamilyEvaluateTradeReq(BaseModel):
    prefer_env_defaults: bool = True
    leaderboard_url: Optional[str] = None
    holdings_e31_url: Optional[str] = None
    holdings_dc_url: Optional[str] = None
    holdings_fe_url: Optional[str] = None

    trade_account: Literal["Easystreet31","DusterCrusher","FinkleIsEinhorn"]
    trade: List[TradeLine]

    multi_subject_rule: Literal["full_each_unique"] = "full_each_unique"
    fragility_mode: Literal["trade_delta","none"] = "trade_delta"
    defend_buffers: Dict[str, int] = Field(default_factory=lambda: {
        "Easystreet31": DEFAULT_DEFEND_BUFFER,
        "DusterCrusher": DEFAULT_DEFEND_BUFFER,
        "FinkleIsEinhorn": DEFAULT_DEFEND_BUFFER
    })
    players_whitelist: Optional[List[str]] = None

class FamilyTradePlusCounterReq(BaseModel):
    prefer_env_defaults: bool = True
    leaderboard_url: Optional[str] = None
    holdings_e31_url: Optional[str] = None
    holdings_dc_url: Optional[str] = None
    holdings_fe_url: Optional[str] = None
    collection_pool_url: Optional[str] = None

    trade_account: Literal["Easystreet31","DusterCrusher","FinkleIsEinhorn"]
    trade: List[TradeLine]

    multi_subject_rule: Literal["full_each_unique"] = "full_each_unique"
    fragility_mode: Literal["trade_delta","none"] = "trade_delta"
    defend_buffers: Dict[str, int] = Field(default_factory=lambda: {
        "Easystreet31": DEFAULT_DEFEND_BUFFER,
        "DusterCrusher": DEFAULT_DEFEND_BUFFER,
        "FinkleIsEinhorn": DEFAULT_DEFEND_BUFFER
    })
    exclude_trade_get_players: bool = True

    thin: int = 15
    upgrade_gap: int = 12
    entry_gap: int = 8
    keep_buffer: int = 30
    scan_top_candidates: int = 60
    max_each: int = 60
    max_multiples_per_card: int = 3

class LeaderboardDeltaReq(BaseModel):
    leaderboard_today_url: str
    leaderboard_yesterday_url: str
    rivals: Optional[List[str]] = None
    min_sp_delta: int = 1

class LeaderboardDeltaEnvReq(BaseModel):
    prefer_env_defaults: bool = True
    leaderboard_today_url: Optional[str] = None
    leaderboard_yesterday_url: Optional[str] = None
    rivals: Optional[List[str]] = None
    min_sp_delta: int = 1

# NEW: internal transfers optimizer
class FamilyTransfersOptimizeReq(BaseModel):
    prefer_env_defaults: bool = True
    leaderboard_url: Optional[str] = None
    holdings_e31_url: Optional[str] = None
    holdings_dc_url: Optional[str] = None
    holdings_fe_url: Optional[str] = None
    collection_e31_url: Optional[str] = None
    collection_dc_url: Optional[str] = None
    collection_fe_url: Optional[str] = None

    defend_buffers: Dict[str, int] = Field(default_factory=lambda: {
        "Easystreet31": DEFAULT_DEFEND_BUFFER,
        "DusterCrusher": DEFAULT_DEFEND_BUFFER,
        "FinkleIsEinhorn": DEFAULT_DEFEND_BUFFER
    })
    max_multiples_per_card: int = 3
    scan_top_candidates: int = 60
    limit_moves: int = 100
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None
    algorithm: Literal["greedy"] = "greedy"  # future: "cp_sat"

# ---------- Routes ----------
@app.get("/defaults")
def get_defaults():
    return {
        "version": APP_VERSION,
        "links": DEFAULT_LINKS,
        "rivals": DEFAULT_RIVALS,
        "defend_buffer_all": DEFAULT_DEFEND_BUFFER
    }

@app.post("/family_evaluate_trade_by_urls")
def family_evaluate_trade_by_urls(req: FamilyEvaluateTradeReq):
    leader = normalize_leaderboard(fetch_xlsx(_pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)))
    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url, req.prefer_env_defaults)

    fam0, per0, det0 = compute_family_qp(leader, accounts)
    alloc, after_accounts = apply_trade_lines_to_accounts(accounts, req.trade, leader, req.trade_account, req.multi_subject_rule)
    fam1, per1, det1 = compute_family_qp(leader, after_accounts)

    trade_players: Set[str] = set()
    for line in req.trade:
        for p in split_multi_subject_players(line.players):
            trade_players.add(p)

    if req.fragility_mode == "trade_delta":
        frag_list = _created_fragile_firsts(det0, det1, req.defend_buffers, restrict_players=trade_players)
        frag_note = (
            "No new fragile firsts created by this trade."
            if len(frag_list) == 0 else
            f"{len(frag_list)} new fragile first(s) created by this trade."
        )
    else:
        frag_list = []
        frag_note = "Fragility check disabled for this request."

    verdict = "ACCEPT" if (fam1 - fam0) > 0 and len(frag_list) == 0 else \
              ("CAUTION" if (fam1 - fam0) >= 0 else "DECLINE")

    return {
        "allocation_plan": alloc,
        "per_account": {
            a: {"qp_before": int(per0[a]), "qp_after": int(per1[a]), "delta_qp": int(per1[a]-per0[a])}
            for a in FAMILY_ACCOUNTS
        },
        "family_qp": {"before": int(fam0), "after": int(fam1), "delta": int(fam1 - fam0)},
        "fragility_alerts": frag_list,
        "fragility_notes": frag_note,
        "verdict": verdict
    }

@app.post("/family_trade_plus_counter_by_urls")
def family_trade_plus_counter_by_urls(req: FamilyTradePlusCounterReq):
    evaluation = family_evaluate_trade_by_urls(FamilyEvaluateTradeReq(
        prefer_env_defaults=req.prefer_env_defaults,
        leaderboard_url=req.leaderboard_url,
        holdings_e31_url=req.holdings_e31_url,
        holdings_dc_url=req.holdings_dc_url,
        holdings_fe_url=req.holdings_fe_url,
        trade_account=req.trade_account,
        trade=req.trade,
        multi_subject_rule=req.multi_subject_rule,
        fragility_mode=req.fragility_mode,
        defend_buffers=req.defend_buffers,
        players_whitelist=None
    ))

    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url, req.prefer_env_defaults)
    leader = normalize_leaderboard(fetch_xlsx(_pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)))
    fam0, _, _ = compute_family_qp(leader, accounts)
    pool_df = parse_collection(fetch_xlsx(_pick_url(req.collection_pool_url, "pool_collection", req.prefer_env_defaults)))
    subset = pool_df.head(req.scan_top_candidates) if req.scan_top_candidates > 0 else pool_df

    trade_get_players: set[str] = set()
    for line in req.trade:
        if line.side == "GET":
            for p in split_multi_subject_players(line.players):
                trade_get_players.add(p)

    picks: List[Dict[str, Any]] = []
    for _, row in subset.iterrows():
        players = split_multi_subject_players(row["players"])
        sp = int(row["sp"]); qty = int(row["qty"])
        if not players or sp <= 0 or qty <= 0:
            continue
        take_max = min(qty, req.max_multiples_per_card)
        best_acct = None; best_gain = 0; best_t = 0
        for acct in FAMILY_ACCOUNTS:
            sim = clone_accounts_sp(accounts)
            for t in range(1, take_max+1):
                apply_card_sp_to_account(sim[acct], players, sp, add=True)
                fam, _, _ = compute_family_qp(leader, sim)
                gain = fam - fam0
                if gain > best_gain:
                    best_gain = int(gain); best_acct = acct; best_t = t
        if best_gain > 0 and best_acct:
            picks.append({
                "card": row["card"], "no": row["no"], "players": players,
                "assign_to": best_acct, "take_n": best_t, "sp": sp, "family_delta_qp": best_gain
            })

    picks.sort(key=lambda x: (-x["family_delta_qp"], str(x["card"])))
    if req.max_each > 0:
        picks = picks[:req.max_each]

    omitted = 0
    if req.exclude_trade_get_players and trade_get_players:
        filtered = []
        for p in picks:
            if any(pp in trade_get_players for pp in p.get("players", [])):
                omitted += 1
                continue
            filtered.append(p)
        picks = filtered

    return {**evaluation, "counter": {"picks": picks, "omitted": omitted}}

# ---------- Leaderboard Delta ----------
@app.post("/leaderboard_delta_by_urls")
def leaderboard_delta_by_urls(req: LeaderboardDeltaReq):
    # Load and normalize sheets
    today = normalize_leaderboard(fetch_xlsx(_sanitize_gsheet_url(req.leaderboard_today_url)))
    yday  = normalize_leaderboard(fetch_xlsx(_sanitize_gsheet_url(req.leaderboard_yesterday_url)))

    # Build strong-key maps (case-insensitive, punctuation/parentheses stripped)
    def to_maps(lst: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, str]]:
        key_to_sp: Dict[str, int] = {}
        key_to_label: Dict[str, str] = {}
        for e in lst:
            raw_user = str(e["user"])
            key = _canon_user_strong(raw_user)
            sp  = _as_int(e["sp"])
            if key not in key_to_sp or sp > key_to_sp[key]:
                key_to_sp[key] = sp
                key_to_label.setdefault(key, raw_user)
        return key_to_sp, key_to_label

    changes = []
    players = sorted(set(list(today.keys()) + list(yday.keys())))
    for p in players:
        t = today.get(p, [])
        y = yday.get(p, [])
        tm, tlabels = to_maps(t)
        ym, ylabels = to_maps(y)

        for key, sp_now in tm.items():
            sp_prev = ym.get(key, 0)
            d = sp_now - sp_prev
            if abs(d) >= req.min_sp_delta:
                label = tlabels.get(key) or ylabels.get(key) or key
                changes.append({"player": p, "user": label, "delta_sp": int(d)})

        t_top = set(_canon_user_strong(e["user"]) for e in t[:3])
        y_top = set(_canon_user_strong(e["user"]) for e in y[:3])
        joined = [tlabels.get(k, k) for k in (t_top - y_top)]
        left   = [ylabels.get(k, k) for k in (y_top - t_top)]
        if joined or left:
            changes.append({"player": p, "top3_joined": joined, "top3_left": left})

    # Rival heatmap
    def to_key(u: str) -> str: return _canon_user_strong(u)
    rivals = [r for r in (req.rivals or DEFAULT_RIVALS)]
    rivals_set = set(to_key(r) for r in rivals)
    heat = Counter()
    for c in changes:
        if "user" in c and to_key(c["user"]) in rivals_set and abs(c.get("delta_sp", 0)) >= req.min_sp_delta:
            heat[c["user"]] += 1
        if "top3_joined" in c:
            for u in c["top3_joined"]:
                if to_key(u) in rivals_set:
                    heat[u] += 1
    rival_heatmap = [{"user": u, "mentions": n} for u, n in heat.most_common(20)]
    return {"changes": changes[:1000], "rival_heatmap": rival_heatmap}

@app.post("/leaderboard_delta")
def leaderboard_delta(req: LeaderboardDeltaEnvReq):
    today_url = _pick_url(req.leaderboard_today_url, "leaderboard", req.prefer_env_defaults)
    yday_url  = _pick_url(req.leaderboard_yesterday_url, "leaderboard_yday", req.prefer_env_defaults)
    return leaderboard_delta_by_urls(LeaderboardDeltaReq(
        leaderboard_today_url=today_url,
        leaderboard_yesterday_url=yday_url,
        rivals=req.rivals,
        min_sp_delta=req.min_sp_delta
    ))

# ---------- NEW: Family Internal Transfers Optimizer (greedy) ----------
def holdings_from_urls(holdings_e31_url: Optional[str], holdings_dc_url: Optional[str],
                       holdings_fe_url: Optional[str], prefer_env_defaults: bool) -> Dict[str, Dict[str, int]]:
    accounts = {a: {} for a in FAMILY_ACCOUNTS}
    url_e31 = _pick_url(holdings_e31_url, "holdings_e31", prefer_env_defaults)
    url_dc  = _pick_url(holdings_dc_url,  "holdings_dc",  prefer_env_defaults)
    url_fe  = _pick_url(holdings_fe_url,  "holdings_fe",  prefer_env_defaults)
    accounts["Easystreet31"]    = parse_holdings(fetch_xlsx(url_e31))
    accounts["DusterCrusher"]   = parse_holdings(fetch_xlsx(url_dc))
    accounts["FinkleIsEinhorn"] = parse_holdings(fetch_xlsx(url_fe))
    return accounts

def collections_from_urls(collection_e31_url: Optional[str], collection_dc_url: Optional[str],
                          collection_fe_url: Optional[str], prefer_env_defaults: bool
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    url_e31 = _pick_url(collection_e31_url, "collection_e31", prefer_env_defaults)
    url_dc  = _pick_url(collection_dc_url,  "collection_dc",  prefer_env_defaults)
    url_fe  = _pick_url(collection_fe_url,  "collection_fe",  prefer_env_defaults)
    out["Easystreet31"]    = parse_collection(fetch_xlsx(url_e31))
    out["DusterCrusher"]   = parse_collection(fetch_xlsx(url_dc))
    out["FinkleIsEinhorn"] = parse_collection(fetch_xlsx(url_fe))
    return out

def _family_had_first_before(player: str, det_before: Dict[str, Dict[str, Any]]) -> bool:
    for a in FAMILY_ACCOUNTS:
        d = det_before.get(a, {}).get(player, {})
        if d.get("rank", 9999) == 1 and (d.get("margin_if_first") is not None):
            return True
    return False

def _family_first_ok_after(player: str,
                           det_after: Dict[str, Dict[str, Any]],
                           buffers: Dict[str, int]) -> bool:
    # After the move, some family account must hold 1st with margin >= its buffer
    holder = None; margin = None
    for a in FAMILY_ACCOUNTS:
        d = det_after.get(a, {}).get(player, {})
        if d.get("rank", 9999) == 1:
            holder = a
            margin = d.get("margin_if_first")
            break
    if holder is None:  # family lost 1st
        return False
    buf = buffers.get(holder, DEFAULT_DEFEND_BUFFER)
    return (margin is not None) and (margin >= buf)

@app.post("/family_internal_transfers_optimize_by_urls")
def family_internal_transfers_optimize_by_urls(req: FamilyTransfersOptimizeReq):
    # Load sources
    leader = normalize_leaderboard(fetch_xlsx(_pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)))
    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url, req.prefer_env_defaults)
    colls = collections_from_urls(req.collection_e31_url, req.collection_dc_url, req.collection_fe_url, req.prefer_env_defaults)

    # Optional scan throttling
    for acct in FAMILY_ACCOUNTS:
        df = colls.get(acct, pd.DataFrame())
        if req.scan_top_candidates > 0 and len(df) > req.scan_top_candidates:
            colls[acct] = df.head(req.scan_top_candidates).copy()

    # Optional whitelist/blacklist filters
    wl = set([p.lower() for p in (req.players_whitelist or [])])
    bl = set([p.lower() for p in (req.players_blacklist or [])])
    def row_ok(players_text: str) -> bool:
        players = [p.lower() for p in split_multi_subject_players(players_text)]
        if wl and not any(p in wl for p in players):
            return False
        if bl and any(p in bl for p in players):
            return False
        return True
    for acct in FAMILY_ACCOUNTS:
        df = colls[acct]
        colls[acct] = df[df["players"].map(row_ok)].reset_index(drop=True)

    # Working state
    fam_now, per_now, det_now = compute_family_qp(leader, accounts)
    plan: List[Dict[str, Any]] = []
    moved_by_row: Dict[Tuple[str, int], int] = defaultdict(int)  # (donor, rowidx) -> moved copies

    # Greedy loop
    while len(plan) < req.limit_moves:
        best_gain = 0
        best_move = None  # (donor, rowidx, receiver, players, sp, card, no)
        for donor in FAMILY_ACCOUNTS:
            df = colls.get(donor, pd.DataFrame())
            for rowidx, row in df.iterrows():
                qty = int(row.get("qty", 0))
                already = moved_by_row[(donor, rowidx)]
                if qty <= already:
                    continue
                if already >= req.max_multiples_per_card:
                    continue
                players = split_multi_subject_players(row.get("players", ""))
                sp = int(row.get("sp", 0))
                if not players or sp <= 0:
                    continue
                for recv in FAMILY_ACCOUNTS:
                    if recv == donor:
                        continue
                    sim_accounts = clone_accounts_sp(accounts)
                    # move ONE copy
                    apply_card_sp_to_account(sim_accounts[donor], players, sp, add=False)
                    apply_card_sp_to_account(sim_accounts[recv],  players, sp, add=True)
                    fam2, per2, det2 = compute_family_qp(leader, sim_accounts)

                    # safety: if family had a 1st on any involved player before, keep 1st with margin >= buffer after
                    safe = True
                    for pl in players:
                        if _family_had_first_before(pl, det_now):
                            if not _family_first_ok_after(pl, det2, req.defend_buffers):
                                safe = False
                                break
                    if not safe:
                        continue

                    gain = fam2 - fam_now
                    if gain > best_gain:
                        best_gain = int(gain)
                        best_move = (donor, rowidx, recv, players, sp, str(row.get("card","")), str(row.get("no","")))
        if not best_move or best_gain <= 0:
            break

        # Apply the best move
        donor, rowidx, recv, players, sp, card_title, card_no = best_move
        apply_card_sp_to_account(accounts[donor], players, sp, add=False)
        apply_card_sp_to_account(accounts[recv],  players, sp, add=True)
        fam_now, per_now, det_now = compute_family_qp(leader, accounts)
        moved_by_row[(donor, rowidx)] += 1

        plan.append({
            "from": donor, "to": recv, "card": card_title, "no": card_no,
            "players": players, "sp_per_copy": sp, "copies": 1, "family_delta_qp": best_gain
        })

    # Consolidate consecutive moves of the same card/from/to
    consolidated: List[Dict[str, Any]] = []
    for step in plan:
        if consolidated and all(step[k]==consolidated[-1][k] for k in ["from","to","card","no","sp_per_copy","players"]):
            consolidated[-1]["copies"] += 1
            consolidated[-1]["family_delta_qp"] += step["family_delta_qp"]
        else:
            consolidated.append(step)

    # Prepare response
    fam0, per0, _ = compute_family_qp(leader, collections_to_zero())  # not meaningful; keep per-account deltas vs start
    # We really want deltas vs pre-optimizer state:
    eval_before, per_before, _ = compute_family_qp(leader, holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url, req.prefer_env_defaults))

    result = {
        "transfer_plan": consolidated,
        "per_account": {
            a: {"qp_before": int(per_before[a]), "qp_after": int(per_now[a]), "delta_qp": int(per_now[a]-per_before[a])}
            for a in FAMILY_ACCOUNTS
        },
        "family_qp": {"before": int(eval_before), "after": int(fam_now), "delta": int(fam_now - eval_before)},
        "constraints": {
            "defend_buffers": req.defend_buffers,
            "max_multiples_per_card": req.max_multiples_per_card,
            "limit_moves": req.limit_moves,
            "scan_top_candidates": req.scan_top_candidates
        },
        "notes": "Greedy optimizer: applies the best +ΔQP move per step, enforcing family-first buffer safety."
    }
    return result

def collections_to_zero() -> Dict[str, Dict[str, int]]:
    # helper used only if needed elsewhere; returns empty holdings
    return {a: {} for a in FAMILY_ACCOUNTS}
