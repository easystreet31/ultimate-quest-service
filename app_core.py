# app_core.py — Ultimate Quest Service (Small-Payload API)
# Version: 4.1.0 (fragility: trade-created-only + notes)
#
# Endpoints (match Actions JSON v4.1.0):
#   • GET  /defaults
#   • POST /family_evaluate_trade_by_urls
#   • POST /family_trade_plus_counter_by_urls
#   • POST /leaderboard_delta_by_urls
#
# Notes:
#   • Accepts Google Sheets "edit/view" links; converts to export XLSX internally.
#   • Fragility is TRADE-ONLY. We report ONLY fragility CREATED by the trade (crossing into fragile).
#   • Adds top‑level "fragility_alerts" (array) PLUS "fragility_notes" (string).

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

APP_VERSION = "4.1.0"
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
    Convert an arbitrary leaderboard workbook into:
      { "<player>": [ {"user": "...", "sp": N}, ... (sorted desc) ] }
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
    sp_col     = next((c for c in df.columns if any(k in c.lower() for k in ["sp","points","subject"])), None)
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if player_col and user_col and sp_col:
        for _, row in df.iterrows():
            p = _norm_player(row.get(player_col, ""))
            u = _canon_user(row.get(user_col, ""))
            sp = _as_int(row.get(sp_col, 0))
            if p and u:
                out[p].append({"user": u, "sp": sp})
    else:
        player_col = next((c for c in df.columns if "player" in c.lower()), df.columns[0])
        account_cols = []
        for c in df.columns:
            cl = c.lower()
            for acct in FAMILY_ACCOUNTS:
                if _canon_user(acct).lower() in cl:
                    account_cols.append((c, None))
        if not account_cols:
            account_cols = [(c, None) for c in df.columns if c != player_col]
        for _, row in df.iterrows():
            p = _norm_player(row.get(player_col, ""))
            if not p:
                continue
            for name_col, sp_col2 in account_cols:
                user = _canon_user(name_col)
                sp = _as_int(row.get(sp_col2 or name_col, 0))
                if user and sp >= 0:
                    out[p].append({"user": user, "sp": sp})
    # Deduplicate to best SP per user and sort
    for p, lst in out.items():
        best: Dict[str, int] = {}
        for e in lst:
            u = e["user"]
            if u not in best or e["sp"] > best[u]:
                best[u] = e["sp"]
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
                best[u] = e["sp"]

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

# --- Fragility helpers ---
def _created_fragile_firsts(
    det_before: Dict[str, Dict[str, Any]],
    det_after: Dict[str, Dict[str, Any]],
    buffers: Dict[str, int],
    restrict_players: Optional[Set[str]] = None
) -> List[str]:
    """
    Return only NEW fragility created by the trade:
      - rank == 1 after AND margin_if_first < buffer
      - was NOT fragile before
      - restricted to players referenced in the trade (if restrict_players is provided)
    """
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

def fragile_firsts(per_player_details: Dict[str, Any], buffer_val: int) -> List[str]:
    # reserved for daily reporting
    frag = []
    for player, d in per_player_details.items():
        if d.get("rank", 9999) == 1:
            margin = d.get("margin_if_first")
            if margin is not None and margin < buffer_val:
                frag.append(f"{player} (margin {int(margin)})")
    return sorted(frag)

def apply_trade_lines_to_accounts(accounts_sp: Dict[str, Dict[str, int]],
                                  trade: List["TradeLine"],
                                  leader: Dict[str, List[Dict[str, Any]]],
                                  trade_account: str,
                                  multi_subject_rule: str = "full_each_unique"
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    alloc_plan: List[Dict[str, Any]] = []
    cur = clone_accounts_sp(accounts_sp)
    fam0, _, _ = compute_family_qp(leader, cur)

    # Allocate GETs greedily by family QP gain
    for line in [l for l in trade if l.side == "GET"]:
        players = split_multi_subject_players(line.players)
        if not players or line.sp <= 0:
            continue
        best_acct = None
        best_gain = -10**9
        best_snapshot = None
        for acct in FAMILY_ACCOUNTS:
            sim = clone_accounts_sp(cur)
            apply_card_sp_to_account(sim[acct], players, line.sp, add=True)
            fam, _, _ = compute_family_qp(leader, sim)
            gain = fam - fam0
            if gain > best_gain:
                best_gain = gain
                best_acct = acct
                best_snapshot = sim
        if best_snapshot is not None:
            cur = best_snapshot
            fam0, _, _ = compute_family_qp(leader, cur)
            alloc_plan.append({"type": "GET", "players": players, "sp": int(line.sp),
                               "to": best_acct, "family_qp_gain": int(best_gain)})

    # Apply GIVEs to the selected trade account
    for line in [l for l in trade if l.side == "GIVE"]:
        players = split_multi_subject_players(line.players)
        if not players or line.sp <= 0:
            continue
        apply_card_sp_to_account(cur[trade_account], players, line.sp, add=False)
        alloc_plan.append({"type": "GIVE", "players": players, "sp": int(line.sp),
                           "from_acct": trade_account})

    return alloc_plan, cur

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

    # Before
    fam0, per0, det0 = compute_family_qp(leader, accounts)

    # Apply trade
    alloc, after_accounts = apply_trade_lines_to_accounts(accounts, req.trade, leader, req.trade_account, req.multi_subject_rule)

    # After
    fam1, per1, det1 = compute_family_qp(leader, after_accounts)

    # Players referenced by the trade (GET or GIVE)
    trade_players: Set[str] = set()
    for line in req.trade:
        for p in split_multi_subject_players(line.players):
            trade_players.add(p)

    # Fragility: only created-by-trade, only for trade players
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
    # Reuse evaluator so fragility is created-only and includes notes
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

    # Counter picks from pool
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

@app.post("/leaderboard_delta_by_urls")
def leaderboard_delta_by_urls(req: LeaderboardDeltaReq):
    today = normalize_leaderboard(fetch_xlsx(_sanitize_gsheet_url(req.leaderboard_today_url)))
    yday  = normalize_leaderboard(fetch_xlsx(_sanitize_gsheet_url(req.leaderboard_yesterday_url)))
    rivals = [_canon_user(r) for r in (req.rivals or DEFAULT_RIVALS)]
    rivals_set = set([r.lower() for r in rivals])

    changes = []
    players = sorted(set(list(today.keys()) + list(yday.keys())))
    for p in players:
        t = today.get(p, [])
        y = yday.get(p, [])
        tm = {_canon_user(e["user"]): e["sp"] for e in t}
        ym = {_canon_user(e["user"]): e["sp"] for e in y}
        for user, sp in tm.items():
            prev = ym.get(user, 0)
            d = sp - prev
            if abs(d) >= req.min_sp_delta:
                changes.append({"player": p, "user": user, "delta_sp": int(d)})
        t_top = set([_canon_user(e["user"]) for e in t[:3]])
        y_top = set([_canon_user(e["user"]) for e in y[:3]])
        joined = list(t_top - y_top)
        left   = list(y_top - t_top)
        if joined or left:
            changes.append({"player": p, "top3_joined": joined, "top3_left": left})

    heat = Counter()
    for c in changes:
        if "user" in c and c["user"].lower() in rivals_set and abs(c.get("delta_sp", 0)) >= req.min_sp_delta:
            heat[c["user"]] += 1
        if "top3_joined" in c:
            for u in c["top3_joined"]:
                if u.lower() in rivals_set:
                    heat[u] += 1
    rival_heatmap = [{"user": u, "mentions": n} for u, n in heat.most_common(20)]
    return {"changes": changes[:1000], "rival_heatmap": rival_heatmap}
