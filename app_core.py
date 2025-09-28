# app_core.py — Ultimate Quest Service (Small-Payload API)
# Version: 4.3.1  (adds per-player "player_changes" + "total_changes" to family trade outputs)
#
# Endpoints (subset shown):
#   • GET  /defaults
#   • POST /family_evaluate_trade_by_urls                  (now includes player_changes, total_changes)
#   • POST /family_trade_plus_counter_by_urls             (carries player_changes, total_changes from evaluation)
#   • POST /leaderboard_delta_by_urls
#   • POST /leaderboard_delta
#   • POST /family_internal_transfers_optimize_by_urls    (if previously deployed)
#
# Notes:
#   • Accepts Google Sheets "edit/view" links; converts to export XLSX internally.
#   • Trades: fragility is TRADE-ONLY; we report ONLY firsts created by the trade (with a friendly note).
#   • Leaderboard delta: robust case-insensitive user matching (parentheses/punctuation stripped).

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

APP_VERSION = "4.3.1"
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
    """ strong canonicalization: strip whitespace, parentheses, punctuation; lowercase """
    if pd.isna(u) or u is None: return ""
    s = str(u)
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Za-z0-9]+", "", s)
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

# ---------- Parsing ----------
def normalize_leaderboard(sheets: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return { player: [ {user: <label>, sp: int}, ... ] }, typically many rows/user entries (long form) or
    many columns/users (wide form). We keep the best SP per user and sort desc.
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
            if not p: continue
            for uc in user_cols:
                out[p].append({"user": uc, "sp": _as_int(row.get(uc, 0))})

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
    df = None
    for name, d in sheets.items():
        if any("player" in c.lower() or "subject" in c.lower() for c in d.columns):
            df = d; break
    if df is None:
        df = list(sheets.values())[0]
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    player_col = next((c for c in df.columns if any(k in c.lower() for k in ["player","subject"])), df.columns[0])
    sp_col = next((c for c in df.columns if any(k in c.lower() for k in ["sp","points"])), None)
    if sp_col is None:
        for c in reversed(df.columns):
            if df[c].dtype != object:
                sp_col = c; break
        sp_col = sp_col or df.columns[-1]
    agg: Dict[str, int] = {}
    for _, row in df.iterrows():
        p = _norm_player(row.get(player_col, ""))
        sp = _as_int(row.get(sp_col, 0))
        if p and sp:
            agg[p] = agg.get(p, 0) + sp
    return agg

def parse_collection(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
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

# ---------- QP math + standings ----------
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
    Notes:
      - margin_if_first here uses leaderboard’s 2nd (fast). For display buffers we compute exact context elsewhere.
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
        last_sp = None; current_rank = 0; seen = 0
        for u, sp in ordered:
            seen += 1
            if last_sp is None or sp < last_sp:
                current_rank = seen
            ranks_for_user[u] = current_rank
            last_sp = sp

        for a in FAMILY_ACCOUNTS:
            r = ranks_for_user.get(a, 9999)
            sp = fam_sp[a]
            per_account_qp[a] += qp_for_rank(r)
            gap_to_first = max(first_sp - sp, 0) if first_sp else 0
            margin_if_first = (sp - (top_sp[1] if len(top_sp) > 1 else 0)) if r == 1 else None
            per_account_details[a][player] = {
                "sp": sp, "rank": r, "gap_to_first": gap_to_first, "margin_if_first": margin_if_first
            }

    family_qp_total = sum(per_account_qp.values())
    return family_qp_total, per_account_qp, per_account_details

def _merged_standings_for_player(player: str,
                                 leader: Dict[str, List[Dict[str, Any]]],
                                 accounts_sp: Dict[str, Dict[str, int]]) -> List[Tuple[str,int]]:
    """
    Build a full standings list for one player: [(label, sp)] sorted desc, dedup by strong key.
    Family account labels are the canonical account names.
    """
    best: Dict[str, Tuple[str,int]] = {}
    # Leaderboard entries
    for e in leader.get(player, []):
        label = str(e["user"])
        key = _canon_user_strong(label)
        sp = _as_int(e["sp"])
        if key not in best or sp > best[key][1]:
            best[key] = (label, sp)
    # Family accounts (override label/key as canonical account)
    for acct in FAMILY_ACCOUNTS:
        sp = _as_int(accounts_sp.get(acct, {}).get(player, 0))
        key = _canon_user_strong(acct)
        if key not in best or sp > best[key][1]:
            best[key] = (acct, sp)   # force label = account name
    ordered = sorted(best.values(), key=lambda x: (-x[1], x[0].lower()))
    return ordered  # [(label, sp), ...]

def _family_ranks_and_buffers(player: str,
                              leader: Dict[str, List[Dict[str, Any]]],
                              accounts_sp: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """
    For a player, compute:
      - rank/sp for each family account
      - best family account (lowest rank) and its buffer_down:
          * if rank 1: sp - sp(2nd)
          * if rank 2: sp - sp(3rd)
          * else: None
      - family_qp for this player (sum over accounts’ ranks)
    """
    standings = _merged_standings_for_player(player, leader, accounts_sp)
    # Build map from label to rank/sp
    label_to_rank: Dict[str, Tuple[int,int]] = {}
    last_sp = None; current_rank = 0; seen = 0
    for idx, (label, sp) in enumerate(standings):
        seen += 1
        if last_sp is None or sp < last_sp:
            current_rank = seen
        label_to_rank[label] = (current_rank, sp)
        last_sp = sp

    # Extract family ranks
    fam_ranks: Dict[str, Tuple[int,int]] = {}
    for acct in FAMILY_ACCOUNTS:
        fam_ranks[acct] = label_to_rank.get(acct, (9999, 0))

    # Determine best family account
    best_acct = min(FAMILY_ACCOUNTS, key=lambda a: (fam_ranks[a][0], -fam_ranks[a][1]))
    best_rank, best_sp = fam_ranks[best_acct]

    # Buffer down
    second_sp = standings[1][1] if len(standings) > 1 else 0
    third_sp  = standings[2][1] if len(standings) > 2 else 0
    buffer_down = None
    if best_rank == 1:
        buffer_down = best_sp - second_sp
    elif best_rank == 2:
        buffer_down = best_sp - third_sp

    # Family QP for this player
    family_qp_player = sum(qp_for_rank(fam_ranks[a][0]) for a in FAMILY_ACCOUNTS)

    return {
        "best_acct": best_acct,
        "best_rank": best_rank,
        "best_sp": best_sp,
        "buffer_down": buffer_down,
        "family_qp_player": family_qp_player,
        "fam_ranks": {a: {"rank": fam_ranks[a][0], "sp": fam_ranks[a][1]} for a in FAMILY_ACCOUNTS}
    }

def _players_touched_by_trade(trade: List["TradeLine"]) -> Set[str]:
    touched: Set[str] = set()
    for line in trade:
        for p in split_multi_subject_players(line.players):
            touched.add(p)
    return touched

def _player_changes_report(players: Set[str],
                           leader: Dict[str, List[Dict[str, Any]]],
                           accounts_before: Dict[str, Dict[str, int]],
                           accounts_after: Dict[str, Dict[str, int]],
                           det_before: Dict[str, Dict[str, Any]],
                           det_after: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str,int]]:
    rows: List[Dict[str, Any]] = []
    tot_sp = 0; tot_buf = 0; tot_qp = 0
    for player in sorted(players):
        # Best family SP before/after (for headline SP change per player)
        sp_b = max(accounts_before.get(a, {}).get(player, 0) for a in FAMILY_ACCOUNTS)
        sp_a = max(accounts_after.get(a, {}).get(player, 0) for a in FAMILY_ACCOUNTS)
        d_sp = sp_a - sp_b

        # Context before/after for best rank & buffer
        ctx_b = _family_ranks_and_buffers(player, leader, accounts_before)
        ctx_a = _family_ranks_and_buffers(player, leader, accounts_after)
        r_b, r_a = ctx_b["best_rank"], ctx_a["best_rank"]
        buf_b, buf_a = ctx_b["buffer_down"], ctx_a["buffer_down"]
        d_buf = None
        if buf_b is not None or buf_a is not None:
            d_buf = (buf_a or 0) - (buf_b or 0)

        # Family QP for this player (sum across family accounts) using the same rank model
        qp_b = ctx_b["family_qp_player"]
        qp_a = ctx_a["family_qp_player"]
        d_qp = qp_a - qp_b

        rows.append({
            "player": player,
            "sp_before": int(sp_b), "sp_after": int(sp_a), "delta_sp": int(d_sp),
            "best_rank_before": int(r_b) if r_b != 9999 else None,
            "best_rank_after": int(r_a) if r_a != 9999 else None,
            "buffer_before": int(buf_b) if buf_b is not None else None,
            "buffer_after": int(buf_a) if buf_a is not None else None,
            "delta_buffer": int(d_buf) if d_buf is not None else None,
            "qp_before": int(qp_b), "qp_after": int(qp_a), "delta_qp": int(d_qp)
        })

        tot_sp += d_sp
        tot_qp += d_qp
        if d_buf is not None:
            tot_buf += d_buf

    totals = {"delta_sp": int(tot_sp), "delta_buffer": int(tot_buf), "delta_qp": int(tot_qp)}
    # Sort players by QP delta desc, then SP delta desc, then name
    rows.sort(key=lambda r: (-r["delta_qp"], -r["delta_sp"], r["player"].lower()))
    return rows, totals

# ---------- Family holdings helpers ----------
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

def parse_collection_pool(prefer_env_defaults: bool, collection_pool_url: Optional[str]) -> pd.DataFrame:
    pool_url = _pick_url(collection_pool_url, "pool_collection", prefer_env_defaults)
    return parse_collection(fetch_xlsx(pool_url))

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

# (Optional) Internal transfers optimizer request if you already merged earlier version
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
    algorithm: Literal["greedy"] = "greedy"

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
    # Load inputs
    leader = normalize_leaderboard(fetch_xlsx(_pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)))
    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url, req.prefer_env_defaults)

    # Evaluate
    fam0, per0, det0 = compute_family_qp(leader, accounts)
    # Allocate GETs greedily by family QP gain, then apply GIVEs on trade_account
    alloc_plan: List[Dict[str, Any]] = []
    cur = {a: dict(v) for a, v in accounts.items()}
    fam_base = fam0

    # Allocate GETs by greedy family-QP gain
    for line in [l for l in req.trade if l.side == "GET"]:
        players = split_multi_subject_players(line.players)
        if not players or line.sp <= 0: continue
        best_acct = None; best_gain = -10**9; best_snapshot = None
        for acct in FAMILY_ACCOUNTS:
            sim = {a: dict(v) for a, v in cur.items()}
            for p in _unique_preserve(players):
                sim[acct][p] = sim[acct].get(p, 0) + line.sp
            fam_sim, _, _ = compute_family_qp(leader, sim)
            gain = fam_sim - fam_base
            if gain > best_gain:
                best_gain = gain; best_acct = acct; best_snapshot = sim
        if best_snapshot is not None:
            cur = best_snapshot
            fam_base, _, _ = compute_family_qp(leader, cur)
            alloc_plan.append({"type": "GET", "players": players, "sp": int(line.sp),
                               "to": best_acct, "family_qp_gain": int(best_gain)})

    # Apply GIVEs to the selected trade account
    for line in [l for l in req.trade if l.side == "GIVE"]:
        players = split_multi_subject_players(line.players)
        if not players or line.sp <= 0: continue
        for p in _unique_preserve(players):
            cur[req.trade_account][p] = cur[req.trade_account].get(p, 0) - line.sp
            if cur[req.trade_account][p] <= 0:
                cur[req.trade_account].pop(p, None)
        alloc_plan.append({"type": "GIVE", "players": players, "sp": int(line.sp),
                           "from_acct": req.trade_account})

    # After-trade family standings
    fam1, per1, det1 = compute_family_qp(leader, cur)

    # Trade-only fragility (created new fragile firsts)
    trade_players = _players_touched_by_trade(req.trade)
    if req.fragility_mode == "trade_delta":
        def _created_fragile_firsts(det_before, det_after, buffers, restrict_players: Optional[Set[str]]):
            alerts: List[str] = []
            for acct in FAMILY_ACCOUNTS:
                buf = buffers.get(acct, DEFAULT_DEFEND_BUFFER)
                after_map = det_after.get(acct, {}) or {}
                before_map = det_before.get(acct, {}) or {}
                for player, ainfo in after_map.items():
                    if restrict_players and player not in restrict_players: continue
                    a_rank = ainfo.get("rank", 9999); a_margin = ainfo.get("margin_if_first")
                    frag_after = (a_rank == 1 and a_margin is not None and a_margin < buf)
                    if not frag_after: continue
                    binfo = before_map.get(player, {})
                    b_rank = binfo.get("rank", 9999); b_margin = binfo.get("margin_if_first")
                    frag_before = (b_rank == 1 and b_margin is not None and b_margin < buf)
                    if frag_after and not frag_before:
                        alerts.append(f"{acct}: {player} (margin {int(a_margin)})")
            return sorted(alerts)
        frag_list = _created_fragile_firsts(det0, det1, req.defend_buffers, restrict_players=trade_players)
        frag_note = ("No new fragile firsts created by this trade."
                     if len(frag_list) == 0 else f"{len(frag_list)} new fragile first(s) created by this trade.")
    else:
        frag_list = []
        frag_note = "Fragility check disabled for this request."

    # NEW: per-player impact + totals (for players touched by the trade)
    player_changes, total_changes = _player_changes_report(
        trade_players, leader, accounts_before=accounts, accounts_after=cur, det_before=det0, det_after=det1
    )

    verdict = "ACCEPT" if (fam1 - fam0) > 0 and len(frag_list) == 0 else \
              ("CAUTION" if (fam1 - fam0) >= 0 else "DECLINE")

    return {
        "allocation_plan": alloc_plan,
        "per_account": {
            a: {"qp_before": int(per0[a]), "qp_after": int(per1[a]), "delta_qp": int(per1[a]-per0[a])}
            for a in FAMILY_ACCOUNTS
        },
        "family_qp": {"before": int(fam0), "after": int(fam1), "delta": int(fam1 - fam0)},
        "player_changes": player_changes,          # <-- NEW
        "total_changes": total_changes,            # <-- NEW
        "fragility_alerts": frag_list,
        "fragility_notes": frag_note,
        "verdict": verdict
    }

@app.post("/family_trade_plus_counter_by_urls")
def family_trade_plus_counter_by_urls(req: FamilyTradePlusCounterReq):
    # Run evaluation first (brings back player_changes + total_changes)
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

    # Counter scan (unchanged)
    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url, req.prefer_env_defaults)
    leader = normalize_leaderboard(fetch_xlsx(_pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)))
    fam0, _, _ = compute_family_qp(leader, accounts)
    pool_df = parse_collection_pool(req.prefer_env_defaults, req.collection_pool_url)
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
        if not players or sp <= 0 or qty <= 0: continue
        take_max = min(qty, req.max_multiples_per_card)
        best_acct = None; best_gain = 0; best_t = 0
        for acct in FAMILY_ACCOUNTS:
            sim = {a: dict(v) for a, v in accounts.items()}
            for t in range(1, take_max+1):
                for p in _unique_preserve(players):
                    sim[acct][p] = sim[acct].get(p, 0) + sp
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
                omitted += 1; continue
            filtered.append(p)
        picks = filtered

    return {**evaluation, "counter": {"picks": picks, "omitted": omitted}}

# ---------- Leaderboard Delta ----------
@app.post("/leaderboard_delta_by_urls")
def leaderboard_delta_by_urls(req: LeaderboardDeltaReq):
    today = normalize_leaderboard(fetch_xlsx(_sanitize_gsheet_url(req.leaderboard_today_url)))
    yday  = normalize_leaderboard(fetch_xlsx(_sanitize_gsheet_url(req.leaderboard_yesterday_url)))

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
        t = today.get(p, []); y = yday.get(p, [])
        tm, tlabels = to_maps(t); ym, ylabels = to_maps(y)

        for key, sp_now in tm.items():
            prev = ym.get(key, 0); d = sp_now - prev
            if abs(d) >= req.min_sp_delta:
                label = tlabels.get(key) or ylabels.get(key) or key
                changes.append({"player": p, "user": label, "delta_sp": int(d)})

        t_top = sorted([(k, tm[k]) for k in tm], key=lambda kv: -kv[1])[:3]
        y_top = sorted([(k, ym[k]) for k in ym], key=lambda kv: -kv[1])[:3]
        t_keys = set(k for k, _ in t_top); y_keys = set(k for k, _ in y_top)
        joined = [tlabels.get(k, k) for k in (t_keys - y_keys)]
        left   = [ylabels.get(k, k) for k in (y_keys - t_keys)]
        if joined or left:
            changes.append({"player": p, "top3_joined": joined, "top3_left": left})

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

# ---------- (Optional) Internal Transfers Optimizer (kept as-is if already merged) ----------
@app.post("/family_internal_transfers_optimize_by_urls")
def family_internal_transfers_optimize_by_urls(req: FamilyTransfersOptimizeReq):
    leader = normalize_leaderboard(fetch_xlsx(_pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)))
    def collections_from_urls(collection_e31_url, collection_dc_url, collection_fe_url, prefer_env_defaults):
        out: Dict[str, pd.DataFrame] = {}
        url_e31 = _pick_url(collection_e31_url, "collection_e31", prefer_env_defaults)
        url_dc  = _pick_url(collection_dc_url,  "collection_dc",  prefer_env_defaults)
        url_fe  = _pick_url(collection_fe_url,  "collection_fe",  prefer_env_defaults)
        out["Easystreet31"]    = parse_collection(fetch_xlsx(url_e31))
        out["DusterCrusher"]   = parse_collection(fetch_xlsx(url_dc))
        out["FinkleIsEinhorn"] = parse_collection(fetch_xlsx(url_fe))
        return out

    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url, req.prefer_env_defaults)
    colls = collections_from_urls(req.collection_e31_url, req.collection_dc_url, req.collection_fe_url, req.prefer_env_defaults)

    for acct in FAMILY_ACCOUNTS:
        df = colls.get(acct, pd.DataFrame())
        if req.scan_top_candidates > 0 and len(df) > req.scan_top_candidates:
            colls[acct] = df.head(req.scan_top_candidates).copy()

    wl = set([p.lower() for p in (req.players_whitelist or [])])
    bl = set([p.lower() for p in (req.players_blacklist or [])])
    def row_ok(players_text: str) -> bool:
        players = [p.lower() for p in split_multi_subject_players(players_text)]
        if wl and not any(p in wl for p in players): return False
        if bl and any(p in bl for p in players): return False
        return True
    for acct in FAMILY_ACCOUNTS:
        df = colls[acct]
        colls[acct] = df[df["players"].map(row_ok)].reset_index(drop=True)

    fam_now, per_now, det_now = compute_family_qp(leader, accounts)
    plan: List[Dict[str, Any]] = []
    moved_by_row: Dict[Tuple[str, int], int] = defaultdict(int)

    while len(plan) < req.limit_moves:
        best_gain = 0; best_move = None
        for donor in FAMILY_ACCOUNTS:
            df = colls.get(donor, pd.DataFrame())
            for rowidx, row in df.iterrows():
                qty = int(row.get("qty", 0)); already = moved_by_row[(donor, rowidx)]
                if qty <= already or already >= req.max_multiples_per_card: continue
                players = split_multi_subject_players(row.get("players", "")); sp = int(row.get("sp", 0))
                if not players or sp <= 0: continue
                for recv in FAMILY_ACCOUNTS:
                    if recv == donor: continue
                    sim_accounts = {a: dict(v) for a, v in accounts.items()}
                    for p in _unique_preserve(players):
                        sim_accounts[donor][p] = sim_accounts[donor].get(p, 0) - sp
                        if sim_accounts[donor][p] <= 0: sim_accounts[donor].pop(p, None)
                        sim_accounts[recv][p]  = sim_accounts[recv].get(p, 0) + sp
                    fam2, per2, det2 = compute_family_qp(leader, sim_accounts)
                    # safety: if family held 1st on a player before, ensure after margin >= buffer
                    safe = True
                    for pl in players:
                        # was any family 1st before?
                        was_first = any((det_now.get(a, {}).get(pl, {}).get("rank", 9999) == 1)
                                        for a in FAMILY_ACCOUNTS)
                        if was_first:
                            # now who holds 1st and with what margin?
                            holder = None; margin = None
                            for a in FAMILY_ACCOUNTS:
                                d = det2.get(a, {}).get(pl, {})
                                if d.get("rank", 9999) == 1:
                                    holder = a; margin = d.get("margin_if_first"); break
                            buf = req.defend_buffers.get(holder, DEFAULT_DEFEND_BUFFER) if holder else DEFAULT_DEFEND_BUFFER
                            if holder is None or margin is None or margin < buf:
                                safe = False; break
                    if not safe: continue
                    gain = fam2 - fam_now
                    if gain > best_gain:
                        best_gain = int(gain)
                        best_move = (donor, rowidx, recv, players, sp, str(row.get("card","")), str(row.get("no","")))
        if not best_move or best_gain <= 0: break
        donor, rowidx, recv, players, sp, card_title, card_no = best_move
        for p in _unique_preserve(players):
            accounts[donor][p] = accounts[donor].get(p, 0) - sp
            if accounts[donor][p] <= 0: accounts[donor].pop(p, None)
            accounts[recv][p]  = accounts[recv].get(p, 0) + sp
        fam_now, per_now, det_now = compute_family_qp(leader, accounts)
        moved_by_row[(donor, rowidx)] += 1
        plan.append({"from": donor, "to": recv, "card": card_title, "no": card_no,
                     "players": players, "sp_per_copy": sp, "copies": 1, "family_delta_qp": best_gain})

    # coalesce consecutive identical steps
    consolidated: List[Dict[str, Any]] = []
    for step in plan:
        if consolidated and all(step[k]==consolidated[-1][k] for k in ["from","to","card","no","sp_per_copy","players"]):
            consolidated[-1]["copies"] += 1
            consolidated[-1]["family_delta_qp"] += step["family_delta_qp"]
        else:
            consolidated.append(step)

    fam_before, per_before, _ = compute_family_qp(leader, holdings_from_urls(None,None,None, True))
    return {
        "transfer_plan": consolidated,
        "per_account": {
            a: {"qp_before": int(per_before[a]), "qp_after": int(per_now[a]), "delta_qp": int(per_now[a]-per_before[a])}
            for a in FAMILY_ACCOUNTS
        },
        "family_qp": {"before": int(fam_before), "after": int(fam_now), "delta": int(fam_now - fam_before)},
        "constraints": {
            "defend_buffers": req.defend_buffers,
            "max_multiples_per_card": req.max_multiples_per_card,
            "limit_moves": req.limit_moves,
            "scan_top_candidates": req.scan_top_candidates
        },
        "notes": "Greedy optimizer: picks the next best +ΔQP move while enforcing family-first buffer safety."
    }
