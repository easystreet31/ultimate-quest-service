# main.py
# Ultimate Quest Service — v3.4.0 "all-in-one + partner-safe-give"
#
# Endpoints
# ---------
# - POST /evaluate_by_urls_easystreet31
# - POST /scan_by_urls_easystreet31
# - POST /scan_rival_by_urls_easystreet31
# - POST /scan_partner_by_urls_easystreet31
# - POST /review_collection_by_urls_easystreet31
# - POST /suggest_give_from_collection_by_urls_easystreet31
# - GET  /health
# - GET  /info
#
# Notes
# -----
# - Accepts Google Sheets XLSX export links (or any .xlsx URL)
# - Robust parsing (fuzzy column names), JSON-safe output (scrubs NaN/Inf)
# - Multi-subject rule: "full_each_unique" (full SP to each distinct player on a multi)
# - Tie handling is conservative: you must EXCEED a competitor’s SP to overtake their rank
# - "me" is Easystreet31 (hard-coded for this service)
#
# Requirements (requirements.txt):
# fastapi==0.115.0
# uvicorn[standard]==0.30.6
# pandas==2.2.2
# openpyxl==3.1.5
# python-multipart==0.0.9
# requests==2.32.3

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import requests
import io
import math
import re

APP_VERSION = "3.4.0-all"

app = FastAPI(title="Ultimate Quest Service", version=APP_VERSION)

# ============================================================
# Utilities
# ============================================================

def _fetch_xlsx(url: str) -> Dict[str, pd.DataFrame]:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = io.BytesIO(resp.content)
    xl = pd.ExcelFile(data)
    out = {}
    for name in xl.sheet_names:
        try:
            df = xl.parse(name)
            if isinstance(df, pd.DataFrame) and not df.empty:
                out[name] = df
        except Exception:
            continue
    if not out:
        raise ValueError("No readable sheets found in XLSX.")
    return out

def _first_sheet_with(df_map: Dict[str, pd.DataFrame], needles: List[str]) -> pd.DataFrame:
    for _, df in df_map.items():
        cols = [str(c).strip().lower() for c in df.columns]
        if all(any(n in c for c in cols) for n in needles):
            return df
    # fallback: first sheet
    return list(df_map.values())[0]

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _norm_user(u: Union[str, float, int]) -> str:
    s = str(u or "").strip()
    # strip trailing " (1234)" if present
    if " (" in s and s.endswith(")"):
        s = s[:s.rfind(" (")].strip()
    return s

def _qp_for_rank(rank: Optional[int]) -> int:
    if rank == 1: return 5
    if rank == 2: return 3
    if rank == 3: return 1
    return 0

def _clean_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _clean_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean_json(v) for v in x]
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        if math.isnan(x) or math.isinf(x):
            return None
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x

def _split_players(players_str: str) -> List[str]:
    if not isinstance(players_str, str):
        return []
    s = players_str.strip()
    if not s:
        return []
    # normalize separators
    for pat in [" and ", ",", "&", "/", "+", ";", "|", "•"]:
        s = s.replace(pat, ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    # unique, preserve order
    seen = set(); out = []
    for p in parts:
        pl = p.lower()
        if pl not in seen:
            out.append(p)
            seen.add(pl)
    return out

# ============================================================
# Parsing
# ============================================================

def parse_leaderboard(url: str) -> pd.DataFrame:
    """
    Expected columns (any close aliases, case-insensitive):
      player / subject / name
      rank
      username / user / collector
      sp / subject_points / points
      (qp optional)
    Keeps only rank <= 5.
    """
    sheets = _fetch_xlsx(url)
    df = _first_sheet_with(sheets, ["player", "rank", "sp"])
    df = _norm_cols(df)

    def pick(cands: List[str]) -> Optional[str]:
        for q in cands:
            for c in df.columns:
                if c == q or c.startswith(q):
                    return c
        return None

    c_player = pick(["player","subject","name"])
    c_rank   = pick(["rank","rnk"])
    c_user   = pick(["username","user","collector","owner"])
    c_sp     = pick(["sp","subject_points","points"])
    c_qp     = pick(["qp","quest_points"])

    if not all([c_player, c_rank, c_user, c_sp]):
        raise ValueError("Leaderboard sheet missing required columns (player, rank, user, sp).")

    out = df[[c_player, c_rank, c_user, c_sp] + ([c_qp] if c_qp else [])].copy()
    out.rename(columns={c_player:"player", c_rank:"rank", c_user:"user", c_sp:"sp"}, inplace=True)
    if c_qp: out.rename(columns={c_qp:"qp"}, inplace=True)
    out["player"] = out["player"].astype(str).str.strip()
    out["user"]   = out["user"].map(_norm_user)
    out["rank"]   = pd.to_numeric(out["rank"], errors="coerce").astype("Int64")
    out["sp"]     = pd.to_numeric(out["sp"], errors="coerce").fillna(0).astype(int)
    if "qp" in out.columns:
        out["qp"] = pd.to_numeric(out["qp"], errors="coerce").fillna(0).astype(int)
    else:
        out["qp"] = out["rank"].map(lambda r: _qp_for_rank(int(r)) if pd.notna(r) else 0)

    out = out[out["rank"].fillna(99) <= 5].reset_index(drop=True)
    return out

def parse_holdings(url: str, me: str = "Easystreet31") -> pd.DataFrame:
    """
    Expected columns (any close aliases):
      player / subject
      you_sp / my_sp / sp
      rank (optional)
      you_qp / qp (optional)
    """
    sheets = _fetch_xlsx(url)
    df = _first_sheet_with(sheets, ["player","sp"])
    df = _norm_cols(df)

    def pick1(cands: List[str]) -> Optional[str]:
        for q in cands:
            for c in df.columns:
                if c == q or c.startswith(q):
                    return c
        return None

    c_player = pick1(["player","subject","name"])
    c_sp     = pick1(["you_sp","my_sp","easystreet31_sp","sp"])
    c_rank   = pick1(["rank","you_rank","my_rank"])
    c_qp     = pick1(["you_qp","my_qp","qp"])

    if not c_player or not c_sp:
        raise ValueError("Holdings sheet missing required columns (player, sp).")

    out = pd.DataFrame()
    out["player"] = df[c_player].astype(str).str.strip()
    out["you_sp"] = pd.to_numeric(df[c_sp], errors="coerce").fillna(0).astype(int)
    out["rank"]   = pd.to_numeric(df[c_rank], errors="coerce").astype("Int64") if c_rank else pd.Series([pd.NA]*len(out))
    out["you_qp"] = pd.to_numeric(df[c_qp], errors="coerce").fillna(0).astype(int) if c_qp else out["rank"].map(lambda r: _qp_for_rank(int(r)) if pd.notna(r) else 0)
    return out.reset_index(drop=True)

def parse_collection(url: str) -> pd.DataFrame:
    """
    Generic collection (seller/partner or my own):
      players/subjects, sp, qty (optional), card_name/title (optional), card_number (optional)
    """
    sheets = _fetch_xlsx(url)
    df = _first_sheet_with(sheets, ["players","sp"])
    df = _norm_cols(df)

    def pick1(cands: List[str]) -> Optional[str]:
        for q in cands:
            for c in df.columns:
                if c == q or c.startswith(q):
                    return c
        return None

    c_players = pick1(["players","player","subjects","subject"])
    c_sp      = pick1(["sp","subject_points","points"])
    c_qty     = pick1(["qty","quantity","count","copies","q"])
    c_card    = pick1(["card_name","card","title"])
    c_no      = pick1(["card_number","card_no","no","number"])

    if not c_players or not c_sp:
        raise ValueError("Collection sheet needs at least 'players' and 'sp' columns.")

    out = pd.DataFrame()
    out["card_name"]   = df[c_card].astype(str).str.strip() if c_card else ""
    out["card_number"] = df[c_no].astype(str).str.strip() if c_no else ""
    out["players_raw"] = df[c_players].astype(str).str.strip()
    out["sp"]          = pd.to_numeric(df[c_sp], errors="coerce").fillna(0).astype(int)
    out["qty"]         = pd.to_numeric(df[c_qty], errors="coerce").fillna(1).astype(int) if c_qty else 1
    return out.reset_index(drop=True)

# ============================================================
# Ranking helpers
# ============================================================

def leaderboard_map(lb: pd.DataFrame) -> Dict[str, List[Tuple[str,int,int]]]:
    d: Dict[str, List[Tuple[str,int,int]]] = {}
    for player, grp in lb.groupby("player"):
        rows = []
        for _, r in grp.sort_values(["rank","sp"], ascending=[True,False]).iterrows():
            rows.append((_norm_user(r["user"]), int(r["sp"]), int(r["rank"])))
        d[player] = rows
    return d

def simulate_insert_rank(arr: List[Tuple[str,int,int]], user: str, new_sp: int) -> Tuple[int, int, List[Tuple[str,int]]]:
    """Insert (user,new_sp) into 'arr' and return (new_rank, margin_or_gap, sorted_list[(user,sp)]).
       Ties keep incumbent ahead; you must exceed SP to pass."""
    base = [(u, sp) for (u, sp, _) in arr]

    # Remove any existing entry of 'user'
    base = [(u, sp) for (u, sp) in base if _norm_user(u).lower() != _norm_user(user).lower()]

    # original order index (for incumbents)
    order_index = {u: i for i, (u, _) in enumerate(base)}
    base.append((user, new_sp))

    def sort_key(item):
        u, sp = item
        idx = order_index.get(u, 10_000_000)  # updated/new users pushed after incumbents on ties
        return (-sp, idx)

    sorted_list = sorted(base, key=sort_key)
    users = [u for (u, _) in sorted_list]
    my_idx = users.index(user)
    new_rank = my_idx + 1

    # margin if first else gap to next better
    if my_idx == 0:
        margin = sorted_list[0][1] - (sorted_list[1][1] if len(sorted_list) > 1 else 0)
    else:
        margin = sorted_list[my_idx-1][1] - sorted_list[my_idx][1]

    return new_rank, margin, sorted_list

def qp_for(user: str, player: str, lb_map: Dict[str, List[Tuple[str,int,int]]], sp: int) -> Tuple[int,int,int,int]:
    """Return (rank, qp, margin_or_gap, second_sp) for user at given SP on this player."""
    arr = lb_map.get(player, [])
    new_rank, mg, new_sorted = simulate_insert_rank(arr, _norm_user(user), sp)
    qp = _qp_for_rank(new_rank)
    second_sp = new_sorted[1][1] if len(new_sorted) > 1 else 0
    return new_rank, qp, mg, second_sp

def ranks_with_two(arr: List[Tuple[str,int,int]], me: str, me_sp: int, rival: str, rival_sp: int) -> Tuple[int,int,int]:
    """Return (my_rank, rival_rank, second_sp) with both me and rival in the list at given SPs."""
    me_n = _norm_user(me); rv_n = _norm_user(rival)
    base = [(u, sp) for (u, sp, _) in arr if _norm_user(u) not in (me_n, rv_n)]
    base.append((me_n, me_sp))
    base.append((rv_n, rival_sp))
    sorted_list = sorted(base, key=lambda t: (-t[1], t[0].lower()))
    ranks = { _norm_user(u): i+1 for i, (u, _) in enumerate(sorted_list) }
    second_sp = sorted_list[1][1] if len(sorted_list) > 1 else 0
    return ranks.get(me_n, 99), ranks.get(rv_n, 99), second_sp

# ============================================================
# Models
# ============================================================

class TradeLine(BaseModel):
    side: str                    # "GET" or "GIVE"
    players: str                 # e.g., "Connor Bedard & Jack Eichel"
    sp: int

class EvaluateReq(BaseModel):
    leaderboard_url: str
    holdings_url: str
    multi_subject_rule: str = "full_each_unique"
    defend_buffer: int = 20
    scope: str = "trade_only"
    max_return_players: int = 120
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None
    trade: List[TradeLine]

class ScanReq(BaseModel):
    leaderboard_url: str
    holdings_url: str
    defend_buffer: int = 20
    upgrade_gap: int = 12
    entry_gap: int = 8
    keep_buffer: int = 30
    max_each: int = 25
    show_all: bool = False
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None

class RivalScanReq(ScanReq):
    focus_rival: str

class PartnerScanReq(BaseModel):
    leaderboard_url: str
    holdings_url: str
    partner: str
    target_rivals: Optional[List[str]] = None
    protect_qp: bool = True
    protect_buffer: int = 20
    max_sp_to_gain: int = 25
    max_each: int = 50
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None

class CollectionReviewReq(BaseModel):
    leaderboard_url: str
    holdings_url: str
    collection_url: str
    multi_subject_rule: str = "full_each_unique"
    focus_rival: Optional[str] = None
    rival_only: bool = False
    max_each: int = 60
    max_multiples_per_card: int = 3
    scan_top_candidates: int = 60
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None
    baseline_trade: Optional[List[TradeLine]] = None

class SafeGiveReq(BaseModel):
    leaderboard_url: str
    holdings_url: str
    my_collection_url: str
    partner: str
    multi_subject_rule: str = "full_each_unique"
    protect_qp: bool = True
    protect_buffer: int = 20
    max_each: int = 50
    max_multiples_per_card: int = 3
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None
    target_rivals: Optional[List[str]] = None
    rival_score_weight: int = 250

# ============================================================
# Core evaluators
# ============================================================

def _players_filter(whitelist: Optional[List[str]], blacklist: Optional[List[str]]):
    if whitelist:
        wl = set(p.lower().strip() for p in whitelist)
        return lambda p: p.lower().strip() in wl
    if blacklist:
        bl = set(p.lower().strip() for p in blacklist)
        return lambda p: p.lower().strip() not in bl
    return lambda p: True

def _apply_trade_to_holdings(my: pd.DataFrame, trade: List[TradeLine], multi_rule: str) -> Tuple[pd.DataFrame, Dict[str,int]]:
    """Return (new_holdings_df, delta_by_player)."""
    deltas: Dict[str,int] = {}
    for t in trade:
        players = _split_players(t.players)
        if multi_rule == "full_each_unique":
            players = list(dict.fromkeys(players))
        else:
            players = players[:1]
        for p in players:
            d = t.sp if t.side.upper() == "GET" else -t.sp
            deltas[p] = deltas.get(p, 0) + d

    my2 = my.set_index("player").copy()
    for p, d in deltas.items():
        if p in my2.index:
            my2.loc[p, "you_sp"] = max(0, int(my2.loc[p, "you_sp"]) + d)
        else:
            my2.loc[p, ["you_sp","rank","you_qp"]] = [max(0, d), pd.NA, 0]
    my2 = my2.reset_index()
    return my2, deltas

def _evaluate_trade(lb: pd.DataFrame, my: pd.DataFrame, trade: List[TradeLine], defend_buffer: int, multi_rule: str,
                    whitelist: Optional[List[str]], blacklist: Optional[List[str]], max_return_players: int):
    me = "Easystreet31"
    lb_map = leaderboard_map(lb)
    allow = _players_filter(whitelist, blacklist)

    my_after, deltas = _apply_trade_to_holdings(my, trade, multi_rule)

    impacts = []
    net_sp = 0
    net_qp = 0
    fragile = []

    # compute before & after for changed players
    for p, d in deltas.items():
        if not allow(p):
            continue
        before_row = my[my["player"] == p]
        you_sp_b = int(before_row["you_sp"].iloc[0]) if not before_row.empty else 0
        rank_b, qp_b, _, sec_b = qp_for(me, p, lb_map, you_sp_b)

        you_sp_a = max(0, you_sp_b + d)
        rank_a, qp_a, _, sec_a = qp_for(me, p, lb_map, you_sp_a)

        net_sp += (you_sp_a - you_sp_b)
        net_qp += (qp_a - qp_b)

        margin_b = (you_sp_b - sec_b) if rank_b == 1 else None
        margin_a = (you_sp_a - sec_a) if rank_a == 1 else None
        if rank_a == 1 and (margin_a is not None) and margin_a < defend_buffer:
            fragile.append(p)

        impacts.append({
            "player": p,
            "you_sp": you_sp_b,
            "you_sp_after": you_sp_a,
            "rank_before": rank_b,
            "rank_after": rank_a,
            "qp_before": qp_b,
            "qp_after": qp_a,
            "margin_before_if_first": margin_b,
            "margin_after_if_first": margin_a
        })

    # sort & cap
    impacts.sort(key=lambda r: (-abs((r["qp_after"]-r["qp_before"])), r["player"]))
    if max_return_players > 0:
        impacts = impacts[:max_return_players]

    overall_sign = "Positive" if net_qp > 0 else ("Negative" if net_qp < 0 else "Neutral")
    verdict = "Accept" if (net_qp > 0 and len(fragile) == 0) else ("Decline" if net_qp <= 0 else "Caution")

    return {
        "player_impacts": impacts,
        "per_account_summaries": [{
            "account": me,
            "net_sp": net_sp,
            "net_qp": net_qp,
            "fragile": fragile
        }],
        "qp_summary": {"overall_net_qp": net_qp, "sign": overall_sign},
        "verdict": verdict
    }

# ============================================================
# Scans
# ============================================================

def _daily_scan(lb: pd.DataFrame, my: pd.DataFrame, defend_buffer: int, upgrade_gap: int, entry_gap: int,
                keep_buffer: int, whitelist: Optional[List[str]], blacklist: Optional[List[str]], max_each: int, show_all: bool):
    me = "Easystreet31"
    lb_map = leaderboard_map(lb)
    allow = _players_filter(whitelist, blacklist)

    thin_firsts = []
    upgrade_opps = []
    top3_entries = []
    overshoots = []
    rival_counts: Dict[str,int] = {}

    # iterate all players we have or appear in leaderboard
    players = sorted(set(my["player"].unique()).union(lb["player"].unique()))
    for p in players:
        if not allow(p): continue
        arr = lb_map.get(p, [])
        first_u, first_sp = (arr[0][0], arr[0][1]) if arr else ("", 0)
        second_sp = arr[1][1] if len(arr) > 1 else 0
        third_sp  = arr[2][1] if len(arr) > 2 else 0

        you_sp = int(my.loc[my["player"]==p, "you_sp"].iloc[0]) if (my["player"]==p).any() else 0
        rank, qp, mg, _ = qp_for(me, p, lb_map, you_sp)

        # thin firsts
        if rank == 1:
            margin = you_sp - second_sp
            if margin <= defend_buffer:
                thin_firsts.append({"player": p, "your": you_sp, "vs": first_u, "first_sp": first_sp, "gap": margin, "add_to_keep": max(0, defend_buffer - margin + 1)})
            if margin > keep_buffer:
                overshoots.append({"player": p, "your": you_sp, "vs": first_u, "first_sp": first_sp, "slack": margin})
        else:
            # upgrade opps: gap to 1st
            gap1 = max(0, first_sp - you_sp + 1)
            if gap1 <= upgrade_gap and gap1 > 0:
                upgrade_opps.append({"player": p, "your": you_sp, "leader": first_u, "leader_sp": first_sp, "gap": gap1, "add_to_take_1st": gap1})
        # enter top3
        if rank > 3:
            gap3 = max(0, third_sp - you_sp + 1)
            if gap3 <= entry_gap and gap3 > 0:
                top3_entries.append({"player": p, "your": you_sp, "third_sp": third_sp, "gap": gap3, "add_to_enter_top3": gap3})

        # rival watchlist counting (who blocks me most)
        if rank != 1 and first_u:
            rival_counts[first_u] = rival_counts.get(first_u, 0) + 1

    def cap(lst):
        if show_all or max_each <= 0: return lst, 0
        return lst[:max_each], max(0, len(lst) - max_each)

    thin_firsts, omit_thin = cap(thin_firsts)
    upgrade_opps, omit_up  = cap(upgrade_opps)
    top3_entries, omit_t3  = cap(top3_entries)
    overshoots, omit_over  = cap(overshoots)

    top_rivals = sorted(rival_counts.items(), key=lambda kv: -kv[1])[:15]
    return {
        "summary": {
            "thin_firsts_count": len(thin_firsts), "omitted_thin": omit_thin,
            "upgrade_opps_count": len(upgrade_opps), "omitted_upgrade": omit_up,
            "top3_entries_count": len(top3_entries), "omitted_top3": omit_t3,
            "overshoots_count": len(overshoots), "omitted_overshoots": omit_over
        },
        "thin_firsts": thin_firsts,
        "upgrade_opps": upgrade_opps,
        "top3_entries": top3_entries,
        "overshoots": overshoots,
        "rival_watchlist": [{"user": u, "mentions": n} for (u, n) in top_rivals]
    }

def _rival_scan(lb: pd.DataFrame, my: pd.DataFrame, focus_rival: str, defend_buffer: int, upgrade_gap: int, entry_gap: int,
                keep_buffer: int, max_each: int, show_all: bool):
    me = "Easystreet31"
    lb_map = leaderboard_map(lb)
    rv = _norm_user(focus_rival)

    threats = []   # where rival is near overtaking me or leading
    opps    = []   # where I can pass the rival quickly
    both_out_top3 = []

    players = sorted(set(my["player"].unique()).union(lb["player"].unique()))
    for p in players:
        arr = lb_map.get(p, [])
        you_sp = int(my.loc[my["player"]==p, "you_sp"].iloc[0]) if (my["player"]==p).any() else 0
        rv_sp = 0
        for (u, sp, _) in arr:
            if _norm_user(u) == rv:
                rv_sp = sp; break

        my_rank, rv_rank, second_sp = ranks_with_two(arr, me, you_sp, rv, rv_sp)
        margin_vs_second = you_sp - second_sp if my_rank == 1 else None

        # threat if I'm first but margin thin vs second (could be rival) OR rival is close to me
        if my_rank == 1:
            if margin_vs_second is not None and margin_vs_second <= defend_buffer:
                threats.append({"player": p, "my_rank": my_rank, "rival_rank": rv_rank, "your": you_sp, "second_sp": second_sp, "margin": margin_vs_second})
        else:
            # opp if I can pass rival within upgrade_gap
            need = max(0, rv_sp - you_sp + 1)
            if need <= upgrade_gap and need > 0:
                opps.append({"player": p, "your": you_sp, "rival_sp": rv_sp, "add_to_beat_rival": need})

        if my_rank > 3 and rv_rank > 3:
            gap3 = max(0, (arr[2][1] if len(arr) > 2 else 0) - you_sp + 1)
            if gap3 <= entry_gap and gap3 > 0:
                both_out_top3.append({"player": p, "your": you_sp, "third_sp": (arr[2][1] if len(arr) > 2 else 0), "add_to_enter_top3": gap3})

    def cap(lst):
        if show_all or max_each <= 0: return lst, 0
        return lst[:max_each], max(0, len(lst) - max_each)

    threats, omit_t = cap(threats)
    opps, omit_o = cap(opps)
    both_out_top3, omit_b = cap(both_out_top3)

    return {
        "focus_rival": rv,
        "threats": threats,
        "opportunities": opps,
        "both_out_top3": both_out_top3,
        "omitted": {"threats": omit_t, "opportunities": omit_o, "both_out_top3": omit_b}
    }

def _partner_scan(lb: pd.DataFrame, my: pd.DataFrame, partner: str, target_rivals: Optional[List[str]],
                  protect_qp: bool, protect_buffer: int, max_sp_to_gain: int, max_each: int,
                  whitelist: Optional[List[str]], blacklist: Optional[List[str]]):
    me = "Easystreet31"
    partner = _norm_user(partner)
    lb_map = leaderboard_map(lb)
    allow = _players_filter(whitelist, blacklist)

    suggestions = []

    players = sorted(set(my["player"].unique()).union(lb["player"].unique()))
    rv_set = set(_norm_user(x) for x in (target_rivals or []))

    for p in players:
        if not allow(p): continue
        arr = lb_map.get(p, [])
        you_sp = int(my.loc[my["player"]==p, "you_sp"].iloc[0]) if (my["player"]==p).any() else 0

        # partner current
        part_sp = 0
        for (u, sp, _) in arr:
            if _norm_user(u) == partner:
                part_sp = sp; break

        # potential QP steps for partner: gain 3rd (QP+1), gain 2nd (QP+3), gain 1st (QP+5)
        # We'll compute minimal SP to reach each target, and enforce protect_qp/buffer for me.
        # Determine thresholds from arr (first/second/third)
        r_first = arr[0][1] if len(arr) > 0 else 0
        r_second = arr[1][1] if len(arr) > 1 else 0
        r_third = arr[2][1] if len(arr) > 2 else 0

        targets = []
        # enter 3rd
        if part_sp < r_third:
            need3 = r_third - part_sp + 1
            targets.append(("enter_top3", 3, need3))
        # move to 2nd
        if part_sp < r_second:
            need2 = r_second - part_sp + 1
            targets.append(("reach_2nd", 2, need2))
        # move to 1st
        if part_sp < r_first:
            need1 = r_first - part_sp + 1
            targets.append(("reach_1st", 1, need1))

        for label, tgt_rank, need in targets:
            if need <= 0 or need > max_sp_to_gain:
                continue
            # Check my protection constraints with partner boosted
            my_rank_after, partner_rank_after, second_sp = ranks_with_two(arr, me, you_sp, partner, part_sp + need)
            my_qp_after = _qp_for_rank(my_rank_after)
            my_rank_now, my_qp_now, _, sec_now = qp_for(me, p, lb_map, you_sp)

            if protect_qp and my_qp_after < my_qp_now:
                continue
            if my_rank_now == 1 and protect_buffer:
                # ensure I keep >= protect_buffer
                # my margin after partner boost is my_sp - new second_sp (which could be partner)
                margin_after = you_sp - second_sp
                if margin_after < protect_buffer:
                    continue

            score = (3 if label == "enter_top3" else 7 if label == "reach_2nd" else 10)
            # Rival scoring bump
            if rv_set:
                # if any rival in top3 gets pushed down by partner's move, bump
                for (u, sp, rk) in arr:
                    if _norm_user(u) in rv_set and rk <= 3 and (part_sp + need) > sp:
                        score += 5

            suggestions.append({
                "player": p,
                "partner_need_sp": need,
                "target_rank": tgt_rank,
                "my_rank_now": my_rank_now,
                "my_rank_after": my_rank_after,
                "protect_ok": True,
                "note": label
            })

    suggestions.sort(key=lambda r: (r["target_rank"], r["partner_need_sp"], r["player"]))
    if max_each > 0:
        omitted = max(0, len(suggestions) - max_each)
        suggestions = suggestions[:max_each]
    else:
        omitted = 0

    return {"partner": partner, "opportunities": suggestions, "omitted": omitted}

# ============================================================
# Collection Review + Safe Give
# ============================================================

def _evaluate_card_take(my_map: Dict[str, Dict[str,int]], lb_map: Dict[str, List[Tuple[str,int,int]]],
                        players: List[str], sp: int, take_n: int, me: str, defend_buffer: int,
                        focus_rival: Optional[str] = None) -> Tuple[int, List[Dict[str,Any]], List[Dict[str,Any]], List[Dict[str,Any]]]:
    """Return (my_total_qp_delta, my_impacts[], rival_impacts[], shore_thin_hits[] if margin improves past buffer)."""
    my_total_qp_delta = 0
    my_impacts = []
    rival_impacts = []
    shore_hits = []
    rv = _norm_user(focus_rival) if focus_rival else None

    for p in players:
        mine = my_map.get(p, {"you_sp":0, "rank":None, "you_qp":0})
        sp_b = int(mine["you_sp"] or 0)
        rank_b, qp_b, _, sec_b = qp_for(me, p, lb_map, sp_b)

        sp_a = sp_b + take_n*sp
        rank_a, qp_a, _, sec_a = qp_for(me, p, lb_map, sp_a)
        my_total_qp_delta += (qp_a - qp_b)

        margin_b = sp_b - sec_b if rank_b == 1 else None
        margin_a = sp_a - sec_a if rank_a == 1 else None
        if margin_b is not None and margin_a is not None and margin_b < defend_buffer <= margin_a:
            shore_hits.append({"player": p, "margin_before": margin_b, "margin_after": margin_a})

        if rv:
            # if we overtake the rival in top-3 as a result, note it (approx)
            arr = lb_map.get(p, [])
            rv_sp = 0
            for (u, s, _) in arr:
                if _norm_user(u) == rv:
                    rv_sp = s; break
            my_rank2, rv_rank2, _ = ranks_with_two(arr, me, sp_a, rv, rv_sp)
            my_rank1, rv_rank1, _ = ranks_with_two(arr, me, sp_b, rv, rv_sp)
            if rv_rank1 <= 3 and my_rank1 > rv_rank1 and my_rank2 <= rv_rank2:
                rival_impacts.append({"player": p, "rival": rv, "note": "overtake rival in top-3"})

        my_impacts.append({
            "player": p,
            "my_sp": sp_b,
            "my_sp_after": sp_a,
            "rank_before": rank_b,
            "rank_after": rank_a,
            "qp_before": qp_b,
            "qp_after": qp_a
        })

    return my_total_qp_delta, my_impacts, rival_impacts, shore_hits

def _review_collection(lb: pd.DataFrame, my: pd.DataFrame, coll: pd.DataFrame, defend_buffer: int,
                       multi_rule: str, focus_rival: Optional[str], rival_only: bool,
                       max_each: int, max_multiples_per_card: int, scan_top_candidates: int,
                       whitelist: Optional[List[str]], blacklist: Optional[List[str]]):
    me = "Easystreet31"
    lb_map = leaderboard_map(lb)
    my_map = {r["player"]: {"you_sp":int(r["you_sp"]), "rank": (int(r["rank"]) if pd.notna(r["rank"]) else None), "you_qp":int(r["you_qp"])}
              for _, r in my.iterrows()}
    allow = _players_filter(whitelist, blacklist)

    qp_positive = []
    shore_thin = []
    take_first = []
    enter_top3 = []
    rival_picks = []

    errors = []

    # Optionally pre-trim candidate cards by whether any player is relevant to us
    candidates = []
    for _, row in coll.iterrows():
        try:
            players = _split_players(str(row.get("players_raw","")))
            if multi_rule == "full_each_unique":
                players = list(dict.fromkeys(players))
            else:
                players = players[:1]
            players = [p for p in players if allow(p)]
            if not players:
                continue
            candidates.append(row)
        except Exception as e:
            errors.append({"card": str(row.get("card_name","")), "error": str(e)})

    if scan_top_candidates > 0:
        candidates = candidates[:scan_top_candidates]

    for _, row in pd.DataFrame(candidates).iterrows():
        card = str(row.get("card_name","") or "(unnamed)")
        no   = str(row.get("card_number","") or "")
        sp   = int(row.get("sp",0) or 0)
        qty  = int(row.get("qty",1) or 1)
        if sp <= 0 or qty <= 0: continue

        players = _split_players(str(row.get("players_raw","")))
        if multi_rule == "full_each_unique":
            players = list(dict.fromkeys(players))
        else:
            players = players[:1]
        players = [p for p in players if allow(p)]
        if not players: continue

        max_n = min(qty, max_multiples_per_card)
        for take_n in range(1, max_n+1):
            try:
                total_qp, my_imp, rv_imp, shore_hits = _evaluate_card_take(my_map, lb_map, players, sp, take_n, me, defend_buffer, focus_rival)
                if focus_rival and rv_imp:
                    rival_picks.append({
                        "card": card, "card_number": no, "players": players, "sp": sp, "take_n": take_n,
                        "my_qp_change": total_qp, "my_impacts": my_imp, "rival_impacts": rv_imp
                    })
                if total_qp > 0 and not rival_only:
                    qp_positive.append({
                        "card": card, "card_number": no, "players": players, "sp": sp, "take_n": take_n,
                        "my_qp_change": total_qp, "my_impacts": my_imp
                    })
                if not rival_only:
                    # classify by goals
                    for m in my_imp:
                        p = m["player"]
                        # recompute before/after for precise gating
                        sp_b = m["my_sp"]; sp_a = m["my_sp_after"]
                        r_b, q_b, _, sec_b = qp_for(me, p, lb_map, sp_b)
                        r_a, q_a, _, sec_a = qp_for(me, p, lb_map, sp_a)
                        if r_b == 1 and (sp_b - sec_b) < defend_buffer and r_a == 1 and (sp_a - sec_a) >= defend_buffer:
                            shore_thin.append({"card": card, "card_number": no, "player": p, "sp": sp, "take_n": take_n,
                                               "margin_before": sp_b-sec_b, "margin_after": sp_a-sec_a})
                        if r_b != 1 and r_a == 1:
                            take_first.append({"card": card, "card_number": no, "player": p, "sp": sp, "take_n": take_n,
                                               "gap_to_first_before": max(0, (lb_map.get(p, [(me,0,1)])[0][1] if lb_map.get(p) else 0) - sp_b + 1)})
                        if r_b > 3 and r_a <= 3:
                            third_sp = (lb_map.get(p, []))[2][1] if len(lb_map.get(p, []))>2 else 0
                            enter_top3.append({"card": card, "card_number": no, "player": p, "sp": sp, "take_n": take_n,
                                               "gap_to_third_before": max(0, third_sp - sp_b + 1)})
            except Exception as e:
                errors.append({"card": card, "take_n": str(take_n), "error": str(e)})

    def cap(lst):
        if max_each <= 0: return lst, 0
        return lst[:max_each], max(0, len(lst) - max_each)

    qp_positive, omit_qp = cap(qp_positive)
    shore_thin, omit_sh = cap(shore_thin)
    take_first, omit_tf = cap(take_first)
    enter_top3, omit_t3 = cap(enter_top3)
    rival_picks, omit_rv = cap(rival_picks)

    return {
        "params": {
            "defend_buffer": defend_buffer,
            "max_each": max_each,
            "max_multiples_per_card": max_multiples_per_card,
            "scan_top_candidates": scan_top_candidates,
            "focus_rival": focus_rival or "",
            "rival_only": rival_only
        },
        "qp_positive_picks": qp_positive,
        "omitted_qp_positive": omit_qp,
        "shore_thin_from_collection": shore_thin,
        "omitted_shore_thin": omit_sh,
        "take_first_from_collection": take_first,
        "omitted_take_first": omit_tf,
        "enter_top3_from_collection": enter_top3,
        "omitted_enter_top3": omit_t3,
        "rival_priority_picks": rival_picks,
        "omitted_rival_priority": omit_rv,
        "diagnostics": {
            "errors_count": len(errors),
            "error_samples": errors[:5],
        }
    }

def _safe_give(lb: pd.DataFrame, my: pd.DataFrame, my_coll: pd.DataFrame, partner: str,
               multi_rule: str, protect_qp: bool, protect_buffer: int, max_each: int,
               max_multiples_per_card: int, whitelist: Optional[List[str]], blacklist: Optional[List[str]],
               target_rivals: Optional[List[str]], rival_score_weight: int):
    me = "Easystreet31"
    partner = _norm_user(partner)
    lb_map = leaderboard_map(lb)
    my_map = {r["player"]: {"you_sp":int(r["you_sp"]), "rank": (int(r["rank"]) if pd.notna(r["rank"]) else None), "you_qp":int(r["you_qp"])}
              for _, r in my.iterrows()}
    allow = _players_filter(whitelist, blacklist)
    rv_set = set(_norm_user(x) for x in (target_rivals or []))

    suggestions = []

    for _, row in my_coll.iterrows():
        card = str(row.get("card_name","") or "(unnamed)")
        no   = str(row.get("card_number","") or "")
        sp   = int(row.get("sp",0) or 0)
        qty  = int(row.get("qty",1) or 1)
        if sp <= 0 or qty <= 0: continue

        players = _split_players(str(row.get("players_raw","")))
        if multi_rule == "full_each_unique":
            players = list(dict.fromkeys(players))
        else:
            players = players[:1]
        players = [p for p in players if allow(p)]
        if not players: continue

        # max copies we can give safely across all involved players
        max_copies_allowed = min(qty, max_multiples_per_card)
        for p in players:
            mine = my_map.get(p, {"you_sp":0,"rank":None,"you_qp":0})
            my_sp = int(mine["you_sp"] or 0)
            my_qp = int(mine["you_qp"] or 0)
            # try giving 0..max copies and keep best "safe" n
            safe_n = 0
            for n in range(0, max_copies_allowed+1):
                my_sp_after = max(0, my_sp - n*sp)
                my_rank_after, my_qp_after, _, sec_after = qp_for(me, p, lb_map, my_sp_after)
                if protect_qp and my_qp_after < my_qp:
                    break
                if my_qp == 5 and protect_buffer:
                    margin_after = my_sp_after - sec_after
                    if margin_after < protect_buffer:
                        break
                safe_n = n
            max_copies_allowed = min(max_copies_allowed, safe_n)

        for take_n in range(1, max_copies_allowed+1):
            my_total_qp_delta = 0
            partner_total_qp_delta = 0
            my_impacts = []
            partner_impacts = []
            rival_impacts = []
            score = 0

            for p in players:
                mine = my_map.get(p, {"you_sp":0,"rank":None,"you_qp":0})
                my_sp_b = int(mine["you_sp"] or 0)
                my_qp_b = int(mine["you_qp"] or 0)
                my_sp_a = max(0, my_sp_b - take_n*sp)
                my_rank_a, my_qp_a, _, _ = qp_for(me, p, lb_map, my_sp_a)
                my_total_qp_delta += (my_qp_a - my_qp_b)
                my_impacts.append({"player": p, "my_sp": my_sp_b, "my_sp_after": my_sp_a,
                                   "qp_before": my_qp_b, "qp_after": my_qp_a})

                # partner before/after
                arr = lb_map.get(p, [])
                part_sp_b = 0
                for (u, s, _) in arr:
                    if _norm_user(u) == partner:
                        part_sp_b = s; break
                pr_b_rank, pr_b_qp, _, _ = qp_for(partner, p, lb_map, part_sp_b)
                part_sp_a = part_sp_b + take_n*sp
                pr_a_rank, pr_a_qp, _, _ = qp_for(partner, p, lb_map, part_sp_a)
                partner_total_qp_delta += (pr_a_qp - pr_b_qp)
                partner_impacts.append({"player": p, "partner_sp": part_sp_b, "partner_sp_after": part_sp_a,
                                        "qp_before": pr_b_qp, "qp_after": pr_a_qp})

                if rv_set and pr_a_qp > pr_b_qp:
                    for (u, s, rk) in arr:
                        if _norm_user(u) in rv_set and rk <= 3 and part_sp_a > s:
                            rival_impacts.append({"player": p, "rival": _norm_user(u), "note": "partner overtakes rival in top-3"})
                            score += rival_score_weight

            if protect_qp and my_total_qp_delta < 0:
                continue
            if partner_total_qp_delta <= 0:
                continue

            score += partner_total_qp_delta * 1000
            suggestions.append({
                "card": card, "card_number": no, "players": players, "sp": sp, "take_n": take_n,
                "my_qp_change": int(my_total_qp_delta), "partner_qp_change": int(partner_total_qp_delta),
                "my_impacts": my_impacts, "partner_impacts": partner_impacts, "rival_impacts": rival_impacts,
                "score": int(score)
            })

    suggestions.sort(key=lambda s: (-s["score"], -s["partner_qp_change"], s["card"], -s["take_n"]))
    omitted = 0
    if max_each > 0:
        omitted = max(0, len(suggestions) - max_each)
        suggestions = suggestions[:max_each]

    return {
        "params": {
            "multi_subject_rule": multi_rule,
            "protect_qp": protect_qp,
            "protect_buffer": protect_buffer,
            "max_each": max_each,
            "max_multiples_per_card": max_multiples_per_card,
            "target_rivals": list(rv_set)
        },
        "safe_gives": suggestions,
        "omitted": omitted
    }

# ============================================================
# Routes
# ============================================================

@app.post("/evaluate_by_urls_easystreet31")
def evaluate_by_urls_easystreet31(req: EvaluateReq):
    try:
        lb = parse_leaderboard(req.leaderboard_url)
        my = parse_holdings(req.holdings_url, me="Easystreet31")
        result = _evaluate_trade(lb, my, req.trade, req.defend_buffer, req.multi_subject_rule,
                                 req.players_whitelist, req.players_blacklist, req.max_return_players)
        return _clean_json(result)
    except Exception as e:
        return {"detail": f"Evaluation failed: {e}"}

@app.post("/scan_by_urls_easystreet31")
def scan_by_urls_easystreet31(req: ScanReq):
    try:
        lb = parse_leaderboard(req.leaderboard_url)
        my = parse_holdings(req.holdings_url, me="Easystreet31")
        result = _daily_scan(lb, my, req.defend_buffer, req.upgrade_gap, req.entry_gap,
                             req.keep_buffer, req.players_whitelist, req.players_blacklist,
                             req.max_each, req.show_all)
        return _clean_json(result)
    except Exception as e:
        return {"detail": f"Daily scan failed: {e}"}

@app.post("/scan_rival_by_urls_easystreet31")
def scan_rival_by_urls_easystreet31(req: RivalScanReq):
    try:
        lb = parse_leaderboard(req.leaderboard_url)
        my = parse_holdings(req.holdings_url, me="Easystreet31")
        result = _rival_scan(lb, my, req.focus_rival, req.defend_buffer, req.upgrade_gap, req.entry_gap,
                             req.keep_buffer, req.max_each, req.show_all)
        return _clean_json(result)
    except Exception as e:
        return {"detail": f"Rival scan failed: {e}"}

@app.post("/scan_partner_by_urls_easystreet31")
def scan_partner_by_urls_easystreet31(req: PartnerScanReq):
    try:
        lb = parse_leaderboard(req.leaderboard_url)
        my = parse_holdings(req.holdings_url, me="Easystreet31")
        result = _partner_scan(lb, my, req.partner, req.target_rivals, req.protect_qp,
                               req.protect_buffer, req.max_sp_to_gain, req.max_each,
                               req.players_whitelist, req.players_blacklist)
        return _clean_json(result)
    except Exception as e:
        return {"detail": f"Partner scan failed: {e}"}

@app.post("/review_collection_by_urls_easystreet31")
def review_collection_by_urls_easystreet31(req: CollectionReviewReq):
    try:
        lb = parse_leaderboard(req.leaderboard_url)
        my = parse_holdings(req.holdings_url, me="Easystreet31")

        # Apply optional baseline trade to my holdings before review (countering a pending swap)
        if req.baseline_trade:
            my, _ = _apply_trade_to_holdings(my, req.baseline_trade, req.multi_subject_rule)

        coll = parse_collection(req.collection_url)
        result = _review_collection(lb, my, coll, req.defend_buffer, req.multi_subject_rule, req.focus_rival,
                                    req.rival_only, req.max_each, req.max_multiples_per_card,
                                    req.scan_top_candidates, req.players_whitelist, req.players_blacklist)
        return _clean_json(result)
    except Exception as e:
        return {"detail": f"Collection review failed: {e}"}

@app.post("/suggest_give_from_collection_by_urls_easystreet31")
def suggest_give_from_collection_by_urls_easystreet31(req: SafeGiveReq):
    try:
        lb = parse_leaderboard(req.leaderboard_url)
        my = parse_holdings(req.holdings_url, me="Easystreet31")
        my_coll = parse_collection(req.my_collection_url)
        result = _safe_give(lb, my, my_coll, req.partner, req.multi_subject_rule, req.protect_qp,
                            req.protect_buffer, req.max_each, req.max_multiples_per_card,
                            req.players_whitelist, req.players_blacklist, req.target_rivals, req.rival_score_weight)
        return _clean_json(result)
    except Exception as e:
        return {"detail": f"Partner safe-give failed: {e}"}

@app.get("/health")
def health():
    return {"ok": True, "version": APP_VERSION}

@app.get("/info")
def info():
    return {
        "version": APP_VERSION,
        "routes": [
            "/docs",
            "/docs/oauth2-redirect",
            "/evaluate_by_urls_easystreet31",
            "/scan_by_urls_easystreet31",
            "/scan_rival_by_urls_easystreet31",
            "/scan_partner_by_urls_easystreet31",
            "/review_collection_by_urls_easystreet31",
            "/suggest_give_from_collection_by_urls_easystreet31",
            "/health",
            "/info"
        ]
    }
