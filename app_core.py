# app_core.py — Ultimate Quest Service (Small-Payload API)
# Version: 4.11.1
#
# 4.10.0 (NEW): All‑In collection simulator
# - New route POST /family_collection_all_in_by_urls
#   Aggregates total SP per player from the seller sheet, then allocates all of it
#   in one pass to maximize family QP; tie-breakers: buffer gain > rank gain > keep holder.
#   This answers: "What is the impact if I own ALL the cards on this collection?"
#
# 4.9.5: GET allocation tie-breakers (ΔQP → buffer → rank → keep-holder)
# 4.9.4: Player-impact buffer/rank computed vs FULL leaderboard using effective SP
#
# Stack: FastAPI + pandas + openpyxl + requests (Python 3.11.9)

from __future__ import annotations
import io, os, re, math, time
from typing import Dict, Any, Optional, List, Tuple, Literal, Set
from collections import defaultdict, Counter

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_VERSION = "4.12.0"
APP_TITLE = "Ultimate Quest Service (Small-Payload API)"

FAMILY_ACCOUNTS = ["Easystreet31", "DusterCrusher", "FinkleIsEinhorn", "UpperDuck"]

# Soft compute budget (ms) — applies to greedy planners (not All‑In)
EVAL_TIME_BUDGET_MS = int(os.getenv("EVAL_TIME_BUDGET_MS", "23000"))

# ---------- Defaults (ENV) ----------
# ---------- Defaults (ENV) ----------
DEFAULT_LINKS = {
    "leaderboard": os.getenv("DEFAULT_LEADERBOARD_URL", ""),
    "leaderboard_yday": os.getenv("DEFAULT_LEADERBOARD_YDAY_URL", ""),
    "holdings_e31": os.getenv("DEFAULT_HOLDINGS_E31_URL", ""),
    "holdings_dc":  os.getenv("DEFAULT_HOLDINGS_DC_URL", ""),
    "holdings_fe":  os.getenv("DEFAULT_HOLDINGS_FE_URL", ""),
    "holdings_ud":  os.getenv("DEFAULT_HOLDINGS_UD_URL", ""),
    "collection_e31": os.getenv("DEFAULT_COLLECTION_E31_URL", ""),
    "collection_dc":  os.getenv("DEFAULT_COLLECTION_DC_URL", ""),
    "collection_fe":  os.getenv("DEFAULT_COLLECTION_FE_URL", ""),
    "collection_ud":  os.getenv("DEFAULT_COLLECTION_UD_URL", ""),
    "pool_collection": os.getenv("DEFAULT_POOL_COLLECTION_URL", ""),
    "player_tags": os.getenv("PLAYER_TAGS_URL", ""),
}
DEFAULT_RIVALS = [u.strip() for u in (os.getenv("DEFAULT_TARGET_RIVALS", "") or
                                      "chfkyle,VjV5,FireRanger,Tfunite,Ovi8").split(",") if u.strip()]
DEFAULT_DEFEND_BUFFER = int(os.getenv("DEFAULT_DEFEND_BUFFER_ALL", "15"))

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# ---------- URL & IO ----------
def _sanitize_gsheet_url(url: str) -> str:
    if not url:
        return url
    try:
        if "docs.google.com/spreadsheets/d/" in url:
            m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
            if m:
                doc_id = m.group(1)
                gid = re.search(r"[?&#]gid=(\d+)", url)
                gid_part = f"&gid={gid.group(1)}" if gid else ""
                return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=xlsx{gid_part}"
    except Exception:
        pass
    return url

def _pick_url(explicit: Optional[str], key: str, prefer_env_defaults: bool) -> str:
    url = (explicit or "").strip()
    if not url:
        url = DEFAULT_LINKS[key]
    return _sanitize_gsheet_url(url)

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
        r = requests.get(url, timeout=60); r.raise_for_status()
        xls = pd.ExcelFile(io.BytesIO(r.content))
        out = {}
        for name in xls.sheet_names:
            df = xls.parse(name)
            df.columns = [str(c).strip() for c in df.columns]
            out[name] = df
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load XLSX from URL: {url} ({e})")

# ---------- Canonicalization ----------
def _norm_player(p: Any) -> str:
    if pd.isna(p) or p is None: return ""
    return re.sub(r"\s+", " ", str(p).strip())

def _canon_user_strong(u: Any) -> str:
    if pd.isna(u) or u is None: return ""
    s = re.sub(r"\(.*?\)", "", str(u))
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Za-z0-9]+", "", s)
    return s.lower()

def _unique_preserve(seq: List[str]) -> List[str]:
    seen = set(); out: List[str] = []
    for s in seq:
        k = s.lower()
        if k not in seen:
            seen.add(k); out.append(s)
    return out

def split_multi_subject_players(players_text: str) -> List[str]:
    if not players_text:
        return []
    parts = re.split(r"\s*(?:/|&|\+|,| and |\||—|–)\s*", str(players_text))
    parts = [_norm_player(p) for p in parts if _norm_player(p)]
    return _unique_preserve(parts)

def _rank_label(r: Optional[int]) -> str:
    if r is None or r >= 9999: return "—"
    if r == 1: return "1st"
    if r == 2: return "2nd"
    if r == 3: return "3rd"
    return f"{int(r)}th"

# ---------- Parse helpers ----------
def normalize_leaderboard(sheets: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
    candidate = None
    for _, df in sheets.items():
        if any("player" in str(c).lower() for c in df.columns):
            candidate = df; break
    if candidate is None:
        candidate = list(sheets.values())[0]
    df = candidate.copy()
    df.columns = [str(c).strip() for c in df.columns]

    player_col = next((c for c in df.columns if "player" in c.lower()), None)
    user_col   = next((c for c in df.columns if any(k in c.lower() for k in ["user","owner","name"])), None)
    sp_col     = next((c for c in df.columns if any(k in c.lower() for k in ["sp","points"])), None)

    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if player_col and user_col and sp_col:
        for _, row in df.iterrows():
            p = _norm_player(row.get(player_col, ""))
            u = str(row.get(user_col, "")).strip()
            sp = _as_int(row.get(sp_col, 0))
            if p and u != "":
                out[p].append({"user": u, "sp": sp})
    else:
        if not player_col:
            player_col = df.columns[0]
        user_cols = [c for c in df.columns if c != player_col]
        for _, row in df.iterrows():
            p = _norm_player(row.get(player_col, ""))
            if not p: continue
            for uc in user_cols:
                out[p].append({"user": uc, "sp": _as_int(row.get(uc, 0))})

    # Dedup/keep max per label
    for p, lst in out.items():
        best: Dict[str, int] = {}
        for e in lst:
            u = str(e["user"]); s = _as_int(e["sp"])
            if u not in best or s > best[u]:
                best[u] = s
        merged = [{"user": u, "sp": best[u]} for u in best]
        merged.sort(key=lambda x: (-x["sp"], str(x["user"]).lower()))
        out[p] = merged
    return out

def parse_holdings(sheets: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    df = None
    for _, d in sheets.items():
        if any(("player" in str(c).lower()) or ("subject" in str(c).lower()) for c in d.columns):
            df = d; break
    if df is None:
        df = list(sheets.values())[0]
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
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

def _series_or_default(df: pd.DataFrame, col: Optional[str], default_value: Any, length: Optional[int] = None) -> pd.Series:
    if col is not None and col in df.columns:
        return df[col]
    n = len(df) if length is None else length
    return pd.Series([default_value] * n, index=df.index if length is None else None)

_TITLE_SPLIT_RE = re.compile(r"\s+(?:—|–|-)\s+")
def _players_from_title(card: str) -> str:
    if not card:
        return ""
    s = str(card).strip()
    parts = _TITLE_SPLIT_RE.split(s)
    tail = parts[-1].strip() if parts else ""
    if "/" in tail: return tail
    if re.search(r"[A-Za-z]\s+[A-Za-z]", tail): return tail
    return ""

def parse_collection(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df_raw = list(sheets.values())[0].copy()
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    col_card = next((c for c in df_raw.columns if any(k in c.lower() for k in ["card","title","item"])), None)
    col_no   = next((c for c in df_raw.columns if c.lower() in ("no","#","num","id","index")), None)
    col_pl   = next((c for c in df_raw.columns if any(k in c.lower() for k in ["players","player","subject"])), None)
    col_sp   = next((c for c in df_raw.columns if any(k in c.lower() for k in ["sp","points","score"])), None)
    col_qty  = next((c for c in df_raw.columns if any(k in c.lower() for k in ["qty","quantity","count","copies"])), None)

    players_s = _series_or_default(df_raw, col_pl, "")
    card_s    = _series_or_default(df_raw, col_card or (df_raw.columns[0] if len(df_raw.columns) else None), "")
    no_s      = _series_or_default(df_raw, col_no, "")
    sp_s      = _series_or_default(df_raw, col_sp, 0).map(_as_int)
    qty_s     = _series_or_default(df_raw, col_qty, 1).map(_as_int)

    out = pd.DataFrame({
        "card": card_s.astype(str).fillna(""),
        "no": no_s.astype(str).fillna(""),
        "players": players_s.astype(str).fillna(""),
        "sp": sp_s.fillna(0).astype(int),
        "qty": qty_s.fillna(1).astype(int),
    })

    mask_empty = (out["players"].str.strip() == "")
    if mask_empty.any():
        out.loc[mask_empty, "players"] = out.loc[mask_empty, "card"].map(_players_from_title).fillna("")
    out["card"] = out["card"].str.strip()
    out["players"] = out["players"].str.strip()
    out = out[(out["card"] != "") & (out["players"] != "")]
    return out.reset_index(drop=True)

# ---------- Ranking ----------
def qp_for_rank(rank: int) -> int:
    return 5 if rank == 1 else 3 if rank == 2 else 1 if rank == 3 else 0

def _smallset_entries_for_player(player: str,
                                 leader: Dict[str, List[Dict[str, Any]]],
                                 accounts_sp: Dict[str, Dict[str, int]]) -> List[Tuple[str, str, int]]:
    rows: List[Tuple[str, str, int]] = []
    for acct in FAMILY_ACCOUNTS:
        sp = _as_int(accounts_sp.get(acct, {}).get(player, 0))
        if sp > 0: rows.append((_canon_user_strong(acct), acct, sp))
    for e in leader.get(player, [])[:3]:
        rows.append((_canon_user_strong(str(e["user"])), str(e["user"]), _as_int(e["sp"])))
    return rows

def _dedup_and_rank(rows: List[Tuple[str, str, int]]) -> Tuple[List[Tuple[str, str, int]], Dict[str, Tuple[int,int]]]:
    best_by_key: Dict[str, Tuple[str, int]] = {}
    for key, label, sp in rows:
        if key not in best_by_key or sp > best_by_key[key][1]:
            best_by_key[key] = (label, sp)
    ordered = sorted(((k, v[0], v[1]) for k, v in best_by_key.items()),
                     key=lambda t: (-t[2], t[1].lower()))
    rank_by_key: Dict[str, Tuple[int,int]] = {}
    last_sp = None; current_rank = 0; seen = 0
    for key, _, sp in ordered:
        seen += 1
        if last_sp is None or sp < last_sp:
            current_rank = seen
        rank_by_key[key] = (current_rank, sp)
        last_sp = sp
    return ordered, rank_by_key

def compute_family_qp(leader: Dict[str, List[Dict[str, Any]]],
                      accounts_sp: Dict[str, Dict[str, int]]
) -> Tuple[int, Dict[str, int], Dict[str, Dict[str, Any]]]:
    per_account_qp: Dict[str, int] = {a: 0 for a in FAMILY_ACCOUNTS}
    per_account_details: Dict[str, Dict[str, Any]] = {a: {} for a in FAMILY_ACCOUNTS}
    all_players = set(leader.keys())
    for a in FAMILY_ACCOUNTS:
        all_players.update(accounts_sp.get(a, {}).keys())
    for player in all_players:
        rows = _smallset_entries_for_player(player, leader, accounts_sp)
        if not rows: continue
        ordered, rank_by_key = _dedup_and_rank(rows)
        first_sp = ordered[0][2] if ordered else 0
        second_sp = ordered[1][2] if len(ordered) > 1 else 0
        for acct in FAMILY_ACCOUNTS:
            key_a = _canon_user_strong(acct)
            r, sp = rank_by_key.get(key_a, (9999, 0))
            per_account_qp[acct] += qp_for_rank(r)
            margin_if_first = (sp - second_sp) if r == 1 else None
            per_account_details[acct][player] = {
                "sp": sp, "rank": r, "gap_to_first": max(first_sp - sp, 0), "margin_if_first": margin_if_first
            }
    family_qp_total = sum(per_account_qp.values())
    return family_qp_total, per_account_qp, per_account_details

def _rank_context_smallset(player: str,
                           leader: Dict[str, List[Dict[str, Any]]],
                           accounts_sp: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    rows = _smallset_entries_for_player(player, leader, accounts_sp)
    if not rows:
        return {"best_acct": None, "best_rank": None, "best_sp": 0,
                "buffer_down": None, "family_qp_player": 0,
                "fam_ranks": {a: {"rank": 9999, "sp": 0} for a in FAMILY_ACCOUNTS}}
    ordered, rank_by_key = _dedup_and_rank(rows)
    best_acct = None; best_rank = 9999; best_sp = 0
    for a in FAMILY_ACCOUNTS:
        r, s = rank_by_key.get(_canon_user_strong(a), (9999, 0))
        if r < best_rank or (r == best_rank and s > best_sp):
            best_acct, best_rank, best_sp = a, r, s
    second_sp = ordered[1][2] if len(ordered) > 1 else 0
    third_sp  = ordered[2][2] if len(ordered) > 2 else 0
    buffer_down = None
    if best_rank == 1: buffer_down = best_sp - second_sp
    elif best_rank == 2: buffer_down = best_sp - third_sp
    family_qp_player = 0; fam_ranks = {}
    for a in FAMILY_ACCOUNTS:
        r, s = rank_by_key.get(_canon_user_strong(a), (9999, 0))
        family_qp_player += qp_for_rank(r)
        fam_ranks[a] = {"rank": r, "sp": s}
    return {"best_acct": best_acct, "best_rank": (best_rank if best_rank != 9999 else None),
            "best_sp": best_sp, "buffer_down": buffer_down,
            "family_qp_player": family_qp_player, "fam_ranks": fam_ranks}

def _rank_and_buffer_full_leader(player: str,
                                 leader: Dict[str, List[Dict[str, Any]]],
                                 fam_sp_map: Dict[str, int]) -> Tuple[Optional[int], Optional[int], Optional[str], int]:
    best_by_key: Dict[str, Tuple[str, int]] = {}
    for e in leader.get(player, []):
        label = str(e["user"]); key = _canon_user_strong(label); sp = _as_int(e["sp"])
        if key not in best_by_key or sp > best_by_key[key][1]:
            best_by_key[key] = (label, sp)
    for acct, sp in fam_sp_map.items():
        if _as_int(sp) <= 0: continue
        key = _canon_user_strong(acct)
        if key not in best_by_key or _as_int(sp) > best_by_key[key][1]:
            best_by_key[key] = (acct, _as_int(sp))
    if not best_by_key:
        return (None, None, None, 0)
    ordered = sorted(((k, v[0], v[1]) for k, v in best_by_key.items()),
                     key=lambda t: (-t[2], t[1].lower()))
    best_rank = 9999; best_sp = 0; best_acct = None
    for idx, (_k, label, sp) in enumerate(ordered, start=1):
        if label in FAMILY_ACCOUNTS:
            if idx < best_rank or (idx == best_rank and sp > best_sp):
                best_rank = idx; best_sp = sp; best_acct = label
    if best_rank >= 9999:
        return (None, None, None, 0)
    cushion = None
    if best_rank == 1:
        second_sp = ordered[1][2] if len(ordered) > 1 else 0
        cushion = best_sp - second_sp
    elif best_rank == 2:
        third_sp = ordered[2][2] if len(ordered) > 2 else 0
        cushion = best_sp - third_sp
    return (best_rank, cushion, best_acct, best_sp)

# ---------- Family holdings ----------
def holdings_from_urls(holdings_e31_url: Optional[str], holdings_dc_url: Optional[str],
                       holdings_fe_url: Optional[str], prefer_env_defaults: bool, holdings_ud_url: Optional[str] = None) -> Dict[str, Dict[str, int]]:
    accounts = {a: {} for a in FAMILY_ACCOUNTS}
    url_e31 = _pick_url(holdings_e31_url, "holdings_e31", prefer_env_defaults)
    url_dc  = _pick_url(holdings_dc_url,  "holdings_dc",  prefer_env_defaults)
    url_fe  = _pick_url(holdings_fe_url,  "holdings_fe",  prefer_env_defaults)
    url_ud  = _pick_url(holdings_ud_url,  "holdings_ud",  prefer_env_defaults)
    accounts["Easystreet31"]    = parse_holdings(fetch_xlsx(url_e31))
    accounts["DusterCrusher"]   = parse_holdings(fetch_xlsx(url_dc))
    accounts["FinkleIsEinhorn"] = parse_holdings(fetch_xlsx(url_fe))
    accounts["UpperDuck"]       = parse_holdings(fetch_xlsx(url_ud)) if url_ud else {}
    return accounts

def parse_collection_pool(prefer_env_defaults: bool, collection_pool_url: Optional[str]) -> pd.DataFrame:
    pool_url = _pick_url(collection_pool_url, "pool_collection", prefer_env_defaults)
    return parse_collection(fetch_xlsx(pool_url))

# ---------- Models ----------
class TradeLine(BaseModel):
    side: Literal["GET","GIVE"]
    players: str
    sp: int

class FamilyEvaluateTradeReq(BaseModel):
    prefer_env_defaults: bool = True
    leaderboard_url: Optional[str] = None
    player_tags_url: Optional[str] = None
    holdings_e31_url: Optional[str] = None
    holdings_dc_url: Optional[str] = None
    holdings_fe_url: Optional[str] = None
    holdings_ud_url: Optional[str] = None
    trade_account: Literal["Easystreet31","DusterCrusher","FinkleIsEinhorn","UpperDuck"]
    trade: List[TradeLine]
    multi_subject_rule: Literal["full_each_unique"] = "full_each_unique"
    fragility_mode: Literal["trade_delta","none"] = "trade_delta"
    defend_buffers: Dict[str, int] = Field(default_factory=lambda: {
        "Easystreet31": DEFAULT_DEFEND_BUFFER,
        "DusterCrusher": DEFAULT_DEFEND_BUFFER,
        "FinkleIsEinhorn": DEFAULT_DEFEND_BUFFER,
        "UpperDuck": DEFAULT_DEFEND_BUFFER
    })
    players_whitelist: Optional[List[str]] = None

class FamilyTradePlusCounterReq(BaseModel):
    prefer_env_defaults: bool = True
    leaderboard_url: Optional[str] = None
    player_tags_url: Optional[str] = None
    holdings_e31_url: Optional[str] = None
    holdings_dc_url: Optional[str] = None
    holdings_fe_url: Optional[str] = None
    holdings_ud_url: Optional[str] = None
    collection_pool_url: Optional[str] = None
    trade_account: Literal["Easystreet31","DusterCrusher","FinkleIsEinhorn","UpperDuck"]
    trade: List[TradeLine]
    multi_subject_rule: Literal["full_each_unique"] = "full_each_unique"
    fragility_mode: Literal["trade_delta","none"] = "trade_delta"
    defend_buffers: Dict[str, int] = Field(default_factory=lambda: {
        "Easystreet31": DEFAULT_DEFEND_BUFFER,
        "DusterCrusher": DEFAULT_DEFEND_BUFFER,
        "FinkleIsEinhorn": DEFAULT_DEFEND_BUFFER,
        "UpperDuck": DEFAULT_DEFEND_BUFFER
    })
    exclude_trade_get_players: bool = True
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None
    scan_top_candidates: int = 0
    max_each: int = 60
    max_multiples_per_card: int = 3
    counter_sp_target: Optional[int] = None
    counter_sp_mode: Literal["closest","at_least","at_most"] = "closest"
    counter_sp_tolerance: int = 0

class FamilyCollectionReviewReq(BaseModel):
    prefer_env_defaults: bool = True
    leaderboard_url: Optional[str] = None
    player_tags_url: Optional[str] = None
    holdings_e31_url: Optional[str] = None
    holdings_dc_url: Optional[str] = None
    holdings_fe_url: Optional[str] = None
    holdings_ud_url: Optional[str] = None
    collection_pool_url: Optional[str] = None
    defend_buffers: Dict[str, int] = Field(default_factory=lambda: {
        "Easystreet31": DEFAULT_DEFEND_BUFFER,
        "DusterCrusher": DEFAULT_DEFEND_BUFFER,
        "FinkleIsEinhorn": DEFAULT_DEFEND_BUFFER,
        "UpperDuck": DEFAULT_DEFEND_BUFFER
    })
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None
    scan_top_candidates: int = 0
    max_each: int = 60
    max_multiples_per_card: int = 3

class FamilyCollectionAllInReq(BaseModel):
    prefer_env_defaults: bool = True
    leaderboard_url: Optional[str] = None
    player_tags_url: Optional[str] = None
    holdings_e31_url: Optional[str] = None
    holdings_dc_url: Optional[str] = None
    holdings_fe_url: Optional[str] = None
    holdings_ud_url: Optional[str] = None
    collection_pool_url: Optional[str] = None
    defend_buffers: Dict[str, int] = Field(default_factory=lambda: {
        "Easystreet31": DEFAULT_DEFEND_BUFFER,
        "DusterCrusher": DEFAULT_DEFEND_BUFFER,
        "FinkleIsEinhorn": DEFAULT_DEFEND_BUFFER,
        "UpperDuck": DEFAULT_DEFEND_BUFFER
    })
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None
    # How to allocate the seller’s entire pool per player
    assign_mode: Literal["best_per_player","to_account"] = "best_per_player"
    assign_to_account: Optional[Literal["Easystreet31","DusterCrusher","FinkleIsEinhorn","UpperDuck"]] = None

class FamilySafeSellReq(BaseModel):
    prefer_env_defaults: bool = True
    leaderboard_url: Optional[str] = None
    player_tags_url: Optional[str] = None
    holdings_e31_url: Optional[str] = None
    holdings_dc_url: Optional[str] = None
    holdings_fe_url: Optional[str] = None
    holdings_ud_url: Optional[str] = None
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None
    include_top3: bool = False
    min_distance_to_rank3: int = 6
    exclude_accounts: Optional[List[Literal["Easystreet31","DusterCrusher","FinkleIsEinhorn","UpperDuck"]]] = None

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

class FamilyFragileWhitelistReq(BaseModel):
    prefer_env_defaults: bool = True
    leaderboard_url: Optional[str] = None
    holdings_e31_url: Optional[str] = None
    holdings_dc_url: Optional[str] = None
    holdings_fe_url: Optional[str] = None
    defend_buffers: Dict[str, int] = Field(default_factory=lambda: {
        "Easystreet31": DEFAULT_DEFEND_BUFFER,
        "DusterCrusher": DEFAULT_DEFEND_BUFFER,
        "FinkleIsEinhorn": DEFAULT_DEFEND_BUFFER
    })
    include_firsts: bool = True
    include_seconds: bool = True
    limit: int = 120

# ---------- Helpers ----------
def _lb_family_sp_for(leader: Dict[str, List[Dict[str, Any]]], player: str, acct: str) -> int:
    key = _canon_user_strong(acct)
    for e in leader.get(player, []):
        if _canon_user_strong(e["user"]) == key:
            return _as_int(e["sp"])
    return 0

# ---------- Tags & Routing ----------
def _load_player_tags(prefer_env_defaults: bool, player_tags_url: Optional[str]) -> Dict[str, Set[str]]:
    tags = {"LEGENDS": set(), "ANA": set(), "DAL": set(), "LAK": set(), "PIT": set()}
    try:
        url = _pick_url(player_tags_url, "player_tags", prefer_env_defaults)
        if not url:
            return tags
        sheets = fetch_xlsx(url)
        def _collect(sheet_name: str, key: str):
            df = None
            if sheet_name in sheets:
                df = sheets[sheet_name]
            else:
                for n, d in sheets.items():
                    if sheet_name.lower() in str(n).lower():
                        df = d; break
            if df is None:
                return
            df = df.copy()
            df.columns = [str(c).strip() for c in df.columns]
            col = df.columns[0]
            for _, row in df.iterrows():
                p = _norm_player(row.get(col, ""))
                if p:
                    tags[key].add(p)
        _collect("Legends", "LEGENDS")
        _collect("ANA", "ANA")
        _collect("DAL", "DAL")
        _collect("LAK", "LAK")
        _collect("PIT", "PIT")
    except Exception:
        pass
    return tags

def _wingnut_headroom_for_player(player: str, leader, accounts_map: Dict[str, Dict[str,int]]) -> int:
    wing = _lb_family_sp_for(leader, player, "Wingnut84")
    fe_eff = max(_lb_family_sp_for(leader, player, "FinkleIsEinhorn"),
                 int(accounts_map.get("FinkleIsEinhorn", {}).get(player, 0)))
    return max((wing - 1) - fe_eff, 0)


def _allowed_accounts_order(players: List[str], leader, accounts_map: Dict[str, Dict[str,int]], tags: Dict[str, Set[str]]) -> List[str]:
    """
    Return a preference-ordered list of family accounts for routing a GET covering `players`.
    Priority:
      - Legends: FE if headroom to trail Wingnut84; else FE last (overflow path)
      - ANA: UD
      - DAL/LAK/PIT: DC
      - No tag: highest-current family holder first (per player), then the rest
    For multi-player lines, preferences are aggregated across players by summing list positions.
    """
    prefs_per_player: List[List[str]] = []
    for p in players:
        p2 = _norm_player(p)
        if p2 in (tags.get("LEGENDS", set()) if tags else set()):
            head = _wingnut_headroom_for_player(p2, leader, accounts_map)
            if head > 0:
                prefs = ["FinkleIsEinhorn","DusterCrusher","Easystreet31","UpperDuck"]
            else:
                # FE cannot take if it would pass Wingnut84 — push FE to the end (overflow)
                prefs = ["DusterCrusher","Easystreet31","UpperDuck","FinkleIsEinhorn"]
        elif p2 in (tags.get("ANA", set()) if tags else set()):
            prefs = ["UpperDuck","Easystreet31","DusterCrusher","FinkleIsEinhorn"]
        elif (p2 in (tags.get("DAL", set()) if tags else set())
              or p2 in (tags.get("LAK", set()) if tags else set())
              or p2 in (tags.get("PIT", set()) if tags else set())):
            prefs = ["DusterCrusher","Easystreet31","FinkleIsEinhorn","UpperDuck"]
        else:
            # Fallback: prefer the current best family holder for this player
            best_acct, best_sp = None, -1
            for a in FAMILY_ACCOUNTS:
                s = int(accounts_map.get(a, {}).get(p2, 0))
                if s > best_sp:
                    best_acct, best_sp = a, s
            if best_acct is None:
                prefs = FAMILY_ACCOUNTS[:]
            else:
                prefs = [best_acct] + [a for a in FAMILY_ACCOUNTS if a != best_acct]
        prefs_per_player.append(prefs)

    scores = {a: 0 for a in FAMILY_ACCOUNTS}
    for prefs in prefs_per_player:
        for idx, a in enumerate(prefs):
            scores[a] += idx
    return sorted(FAMILY_ACCOUNTS, key=lambda a: (scores.get(a, 9999), FAMILY_ACCOUNTS.index(a)))
def _simulate_family_trade_allocation(leader, accounts_in: Dict[str, Dict[str, int]],
                                      trade: List[TradeLine], tags: Optional[Dict[str, Set[str]]] = None) -> Tuple[Dict[str, Dict[str, int]], List[Dict[str, Any]]]:
    cur = {a: dict(v) for a, v in accounts_in.items()}
    fam_base, _, _ = compute_family_qp(leader, cur)
    alloc_plan: List[Dict[str, Any]] = []

    def _eff_map_for(player: str, base_map: Dict[str, Dict[str,int]]) -> Dict[str,int]:
        return {
            acct: max(_lb_family_sp_for(leader, player, acct),
                      int(base_map.get(acct, {}).get(player, 0)))
            for acct in FAMILY_ACCOUNTS
        }

    for line in [l for l in trade if l.side == "GET"]:
        players = split_multi_subject_players(line.players)
        if not players or line.sp <= 0:
            continue
        best_tuple = None; best_snapshot = None; best_meta = None
        
        # Candidate accounts based on tags/rolebook (tags-first; else best-current-holder fallback)
        order = _allowed_accounts_order(players, leader, cur, tags or {"LEGENDS":set(),"ANA":set(),"DAL":set(),"LAK":set(),"PIT":set()})
        # Detect if any player in this line carries a tag
        has_tag = any((_norm_player(p) in (tags or {}).get("LEGENDS", set()) or
                       _norm_player(p) in (tags or {}).get("ANA", set()) or
                       _norm_player(p) in (tags or {}).get("DAL", set()) or
                       _norm_player(p) in (tags or {}).get("LAK", set()) or
                       _norm_player(p) in (tags or {}).get("PIT", set()))
                       for p in players)
        acct_candidates = [order[0]] if (order and has_tag) else (order or FAMILY_ACCOUNTS)
        for acct in acct_candidates:

            sim = {a: dict(v) for a, v in cur.items()}
            for p in _unique_preserve(players):
                sim[acct][p] = sim[acct].get(p, 0) + line.sp
            fam_sim, _, _ = compute_family_qp(leader, sim)
            fam_gain = fam_sim - fam_base

            sum_buf_gain = 0; sum_rank_gain = 0; keep_holder_votes = 0
            for p in players:
                eff_b = _eff_map_for(p, cur); eff_a = _eff_map_for(p, sim)
                r_b, buf_b, best_b_acct, _ = _rank_and_buffer_full_leader(p, leader, eff_b)
                r_a, buf_a, _best_a_acct, _ = _rank_and_buffer_full_leader(p, leader, eff_a)
                buf_gain = 0 if (buf_b is None and buf_a is None) else max(0, (buf_a or 0)-(buf_b or 0))
                sum_buf_gain += buf_gain
                rank_sc_b = 0 if (r_b is None) else -r_b
                rank_sc_a = 0 if (r_a is None) else -r_a
                sum_rank_gain += max(0, (rank_sc_a - rank_sc_b))
                if best_b_acct is not None and acct == best_b_acct:
                    keep_holder_votes += 1
            score = (int(fam_gain), int(sum_buf_gain), int(sum_rank_gain), int(keep_holder_votes))
            if (best_tuple is None) or (score > best_tuple):
                best_tuple, best_snapshot = score, sim
                best_meta = {"to": acct, "players": players, "sp": int(line.sp), "family_qp_gain": int(fam_gain),
                             "routing_trace": {"policy": ("tag_first" if has_tag else "best_holder_fallback"), "order": order},
                             "score_breakdown": {"buffer_gain": int(sum_buf_gain),
                                                 "rank_gain": int(sum_rank_gain),
                                                 "keep_holder_votes": int(keep_holder_votes)}}
        if best_snapshot is not None:
            cur = best_snapshot
            fam_base, _, _ = compute_family_qp(leader, cur)
            alloc_plan.append({"type": "GET", **best_meta})

    return cur, alloc_plan

# ---------- Greedy buyer (unchanged) ----------
def _row_ok(players_text: str, wl: Set[str], bl: Set[str]) -> bool:
    s = "" if players_text is None else str(players_text)
    players = [p.lower() for p in split_multi_subject_players(s)]
    if wl and not any(p in wl for p in players): return False
    if bl and any(p in bl for p in players): return False
    return True

def _apply_card(accounts: Dict[str, Dict[str, int]], acct: str, players: List[str], sp: int, t: int):
    for p in _unique_preserve(players):
        accounts[acct][p] = accounts[acct].get(p, 0) + sp * t

def _competitor_levels_for(acct: str, player: str, leader, accounts_now):
    rows = _smallset_entries_for_player(player, leader, accounts_now)
    if not rows: return (0, 0, 0)
    ordered, _ = _dedup_and_rank(rows)
    key_acct = _canon_user_strong(acct)
    filtered = [(k, sp) for (k, _, sp) in ordered if k != key_acct]
    vals = [sp for _, sp in filtered] + [0, 0, 0]
    return vals[0], vals[1], vals[2]

def _candidate_t_values(acct: str, players: List[str], sp_per_copy: int, avail: int,
                        leader, accounts_now, defend_buffers: Dict[str,int], tags: Optional[Dict[str, Set[str]]] = None) -> List[int]:
    if avail <= 0 or sp_per_copy <= 0:
        return []
    tset: Set[int] = set([1])
    for pl in players:
        s0 = int(accounts_now.get(acct, {}).get(pl, 0))
        t1, t2, t3 = _competitor_levels_for(acct, pl, leader, accounts_now)
        need_top3 = max((t3 + 1) - s0, 0)
        need_2nd  = max((t2 + 1) - s0, 0)
        need_1st  = max((t1 + 1) - s0, 0)
        buf_target = int(defend_buffers.get(acct, DEFAULT_DEFEND_BUFFER))
        need_buf   = max((t2 + buf_target) - s0, 0)
        for need in (need_top3, need_2nd, need_1st, need_buf):
            if need > 0:
                t = (need + sp_per_copy - 1) // sp_per_copy
                if 1 <= t <= avail:
                    tset.add(t)
    tset.add(avail)
    candidates = sorted(tset)
    if len(candidates) > 6:
        candidates = sorted(set(candidates[:5] + [avail]))
    return candidates

def _budget_satisfied(total_sp: int, tgt: Optional[int], mode: str, tol: int) -> bool:
    if tgt is None: return False
    mode = (mode or "closest").lower()
    tol = max(0, int(tol or 0))
    if mode == "at_most":
        return (total_sp <= tgt) and ((tgt - total_sp) <= tol)
    return (total_sp >= tgt) and ((total_sp - tgt) <= tol)

def _plan_collection_buys_greedy(
    leader, accounts_start: Dict[str, Dict[str, int]], pool_df: pd.DataFrame,
    defend_buffers: Dict[str, int], scan_top_candidates: int, max_each: int, max_multiples_per_card: int,
    players_whitelist: Optional[List[str]] = None, players_blacklist: Optional[List[str]] = None,
    ignore_players: Optional[Set[str]] = None,
    budget_target: Optional[int] = None, budget_mode: str = "closest", budget_tolerance: int = 0,
    tags: Optional[Dict[str, Set[str]]] = None
):
    start = time.monotonic()
    def over_budget_time() -> bool:
        return (time.monotonic() - start) * 1000.0 > EVAL_TIME_BUDGET_MS

    ignore_players = set([p.lower() for p in (ignore_players or set())])
    wl = set([p.lower() for p in (players_whitelist or [])])
    bl = set([p.lower() for p in (players_blacklist or [])])

    df = pool_df.copy()
    if scan_top_candidates > 0 and len(df) > scan_top_candidates:
        df = df.head(scan_top_candidates).copy()
    df = df[df["players"].map(lambda s: _row_ok(s, wl, bl))].reset_index(drop=True)

    fam0, _, _ = compute_family_qp(leader, accounts_start)
    accounts_now = {a: dict(v) for a, v in accounts_start.items()}
    fam_now = fam0

    used: Dict[int, int] = defaultdict(int)
    plan: List[Dict[str, Any]] = []
    sims = 0
    total_sp_purchased = 0

    if max_each <= 0 or len(df) == 0:
        fam_after, _, _ = compute_family_qp(leader, accounts_now)
        return {
            "picks": [],
            "summary": {
                "family_qp": {"before": int(fam0), "after": int(fam_after), "delta": int(fam_after - fam0)},
                "counts": {"considered_rows": int(len(df)), "with_players": int(len(df)), "selected": 0, "simulations": sims},
                "sp_budget": {"target": budget_target, "mode": budget_mode, "tolerance": int(budget_tolerance), "purchased_sp": int(total_sp_purchased), "satisfied": False},
                "partial": False
            }
        }

    while len(plan) < max_each:
        if over_budget_time(): break
        best = None
        for idx, row in df.iterrows():
            if over_budget_time(): break
            qty = int(_as_int(row.get("qty", 0)))
            sp  = int(_as_int(row.get("sp", 0)))
            players = split_multi_subject_players(row.get("players", ""))
            if not players or sp <= 0: continue
            if any(p.lower() in (ignore_players or set()) for p in players): continue

            avail = min(max(qty - used[idx], 0), max_multiples_per_card)
            if avail <= 0: continue

            order = _allowed_accounts_order(players, leader, accounts_now, (tags or {"LEGENDS":set(),"ANA":set(),"DAL":set(),"LAK":set(),"PIT":set()}))
            for acct in order:
                t_candidates = _candidate_t_values(acct, players, sp, avail, leader, accounts_now, defend_buffers, tags)
                for t in t_candidates:
                    if over_budget_time(): break
                    sim = {a: dict(v) for a, v in accounts_now.items()}
                    _apply_card(sim, acct, players, sp, t)
                    fam2, _, _ = compute_family_qp(leader, sim)
                    gain = fam2 - fam_now
                    sims += 1

                    primary = None
                    best_rank_jump = -999
                    worst_buffer_after = None
                    for pl in players:
                        ctx_b = _rank_context_smallset(pl, leader, accounts_now)
                        ctx_a = _rank_context_smallset(pl, leader, sim)
                        r_b = ctx_b["best_rank"] or 9999
                        r_a = ctx_a["best_rank"] or 9999
                        buf_b = ctx_b["buffer_down"]; buf_a = ctx_a["buffer_down"]
                        rank_jump = (0 if r_b == 9999 else -r_b) - (0 if r_a == 9999 else -r_a)
                        if rank_jump > best_rank_jump:
                            best_rank_jump = rank_jump
                            primary = {"player": pl,
                                       "rank_before": None if r_b == 9999 else int(r_b),
                                       "rank_after":  None if r_a == 9999 else int(r_a),
                                       "buffer_before": None if buf_b is None else int(buf_b),
                                       "buffer_after":  None if buf_a is None else int(buf_a)}
                            worst_buffer_after = buf_a
                        elif rank_jump == best_rank_jump:
                            if (buf_a or 10**9) < (worst_buffer_after or 10**9):
                                worst_buffer_after = buf_a
                                primary = {"player": pl,
                                           "rank_before": None if r_b == 9999 else int(r_b),
                                           "rank_after":  None if r_a == 9999 else int(r_a),
                                           "buffer_before": None if buf_b is None else int(buf_b),
                                           "buffer_after":  None if buf_a is None else int(buf_a)}
                    if primary is None: continue

                    if gain > 0:
                        score = (1, int(gain), -int(primary.get("buffer_after") if primary.get("buffer_after") is not None else 10**6))
                        category = "QP_GAIN"
                    else:
                        rank_after = primary["rank_after"] or 9999
                        buffer_after = primary["buffer_after"]; buffer_before = primary["buffer_before"]
                        holder_after = _rank_context_smallset(primary["player"], leader, sim)["best_acct"]
                        if rank_after in (1, 2) and holder_after == acct and (buffer_after or 0) > (buffer_before or 0):
                            score = (0, 0, -int(buffer_after if buffer_after is not None else -10**6))
                            category = "BUFFER_SHORE"
                        else:
                            continue

                    cand = {"row_idx": int(idx), "from_qty": int(used[idx]), "take_n": int(t),
                            "assign_to": acct, "players": players, "card": str(row.get("card","")),
                            "no": str(row.get("no","")), "sp": int(sp), "family_delta_qp": int(gain),
                            "category": category, "primary_player": primary["player"],
                            "rank_before": primary["rank_before"], "rank_after": primary["rank_after"],
                            "buffer_before": primary["buffer_before"], "buffer_after": primary["buffer_after"],
                            "holder_after": _rank_context_smallset(primary["player"], leader, sim)["best_acct"]}
                    if (best is None) or (score > best[0]):
                        best = (score, cand, sim, fam2)

        if not best: break
        _, cand, sim_after, fam2 = best
        used[cand["row_idx"]] += cand["take_n"]
        accounts_now = sim_after
        fam_now = fam2
        total_sp_purchased += int(cand["sp"]) * int(cand["take_n"])

        holder_after = cand["holder_after"]
        buf_target = defend_buffers.get(holder_after or cand["assign_to"], DEFAULT_DEFEND_BUFFER)
        if cand["rank_after"] in (1, 2) and cand["buffer_after"] is not None:
            left = max(buf_target - cand["buffer_after"], 0)
            copies_more = math.ceil(left / max(cand["sp"], 1)) if left > 0 else 0
        else:
            left = None; copies_more = None

        cand["buffer_target"] = int(buf_target) if cand["rank_after"] in (1,2) else None
        cand["copies_needed_for_threshold"] = int(copies_more) if copies_more is not None else None

        rb, ra = cand["rank_before"], cand["rank_after"]
        if cand["category"] == "QP_GAIN":
            if (rb or 9999) > 3 and (ra or 9999) <= 3: note = "enter top‑3"
            elif rb == 3 and ra == 2: note = "to 2nd"
            elif rb == 2 and ra == 1: note = "to 1st"
            elif rb == 3 and ra == 1: note = "to 1st (from 3rd)"
            else: note = "QP gain"
        else:
            note = f"buffer +{cand['buffer_after']} total (target {cand['buffer_target']})" if cand["buffer_after"] is not None and cand["buffer_target"] is not None else "buffer add"
        cand["note"] = note

        plan.append(cand)
        if _budget_satisfied(total_sp_purchased, budget_target, budget_mode, budget_tolerance): break

    fam_before, _, _ = compute_family_qp(leader, accounts_start)
    fam_after, _, _ = compute_family_qp(leader, accounts_now)
    with_players = int(len(df))
    return {
        "picks": plan,
        "summary": {
            "family_qp": {"before": int(fam_before), "after": int(fam_after), "delta": int(fam_after - fam_before)},
            "counts": {"considered_rows": int(len(df)), "with_players": with_players, "selected": int(len(plan)), "simulations": sims},
            "sp_budget": {"target": budget_target, "mode": budget_mode, "tolerance": int(budget_tolerance),
                          "purchased_sp": int(total_sp_purchased),
                          "satisfied": _budget_satisfied(total_sp_purchased, budget_target, budget_mode, budget_tolerance) if budget_target is not None else False},
            "partial": (len(plan) < max_each and ((time.monotonic() - start) * 1000.0 > EVAL_TIME_BUDGET_MS))
        }
    }

# ---------- ROUTES ----------
@app.get("/healthz")
def healthz():
    return {"ok": True, "version": APP_VERSION}

@app.get("/defaults")
def get_defaults():
    return {
        "version": APP_VERSION,
        "links": DEFAULT_LINKS,
        "rivals": DEFAULT_RIVALS,
        "defend_buffer_all": DEFAULT_DEFEND_BUFFER
    }

# ---- Family evaluate trade
@app.post("/family_evaluate_trade_by_urls")
def family_evaluate_trade_by_urls(req: FamilyEvaluateTradeReq):
    tags_map = _load_player_tags(req.prefer_env_defaults, req.player_tags_url)
    leader = normalize_leaderboard(fetch_xlsx(_pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)))
    accounts_before = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url, req.prefer_env_defaults, req.holdings_ud_url)
    fam0, per0, det0 = compute_family_qp(leader, accounts_before)

    give_requested: Dict[str, int] = defaultdict(int)
    for line in [l for l in req.trade if l.side == "GIVE"]:
        for p in split_multi_subject_players(line.players):
            give_requested[p] += int(line.sp)

    cur, alloc_plan = _simulate_family_trade_allocation(leader, accounts_before, req.trade, tags_map)

    # Apply GIVEs to trade_account
    for line in [l for l in req.trade if l.side == "GIVE"]:
        players = split_multi_subject_players(line.players)
        if not players or line.sp <= 0: continue
        for p in _unique_preserve(players):
            cur[req.trade_account][p] = cur[req.trade_account].get(p, 0) - line.sp
            if cur[req.trade_account][p] <= 0:
                cur[req.trade_account].pop(p, None)

    fam1, per1, det1 = compute_family_qp(leader, cur)

    touched = set()
    for tl in req.trade:
        for pl in split_multi_subject_players(tl.players):
            touched.add(pl)

    # Ownership warnings vs pre-trade holdings
    ownership_warnings = []
    for p, want in sorted(give_requested.items(), key=lambda kv: kv[0].lower()):
        have = int(accounts_before.get(req.trade_account, {}).get(p, 0))
        if want > have:
            ownership_warnings.append({
                "account": req.trade_account, "player": p,
                "attempted_give_sp": int(want), "owned_sp_before": int(have),
                "note": "GIVE exceeds owned SP; family total may not drop as expected."
            })

    # Player changes — effective SP against FULL leaderboard
    rows=[]; tot_buf=0
    for pl in sorted(touched):
        d_by_acct = {a: int(cur.get(a, {}).get(pl, 0) - accounts_before.get(a, {}).get(pl, 0)) for a in FAMILY_ACCOUNTS}
        lb_base = {a: _lb_family_sp_for(leader, pl, a) for a in FAMILY_ACCOUNTS}
        hold_b = {a: int(accounts_before.get(a, {}).get(pl, 0)) for a in FAMILY_ACCOUNTS}
        eff_b = {a: max(lb_base[a], hold_b[a]) for a in FAMILY_ACCOUNTS}
        eff_a = {a: eff_b[a] + d_by_acct[a] for a in FAMILY_ACCOUNTS}
        sp_b = sum(eff_b.values()); sp_a = sp_b + int(inc_sp); d_sp = int(inc_sp)
        eff_before_map = {a: {pl: eff_b[a]} for a in FAMILY_ACCOUNTS}
        eff_after_map  = {a: {pl: eff_a[a]} for a in FAMILY_ACCOUNTS}
        ctx_b = _rank_context_smallset(pl, leader, eff_before_map)
        ctx_a = _rank_context_smallset(pl, leader, eff_after_map)
        d_qp = ctx_a["family_qp_player"] - ctx_b["family_qp_player"]
        r_b_full, buf_b_full, _, _ = _rank_and_buffer_full_leader(pl, leader, {a: eff_b[a] for a in FAMILY_ACCOUNTS})
        r_a_full, buf_a_full, _, _ = _rank_and_buffer_full_leader(pl, leader, {a: eff_a[a] for a in FAMILY_ACCOUNTS})
        d_buf = None if (buf_b_full is None and buf_a_full is None) else ( (buf_a_full or 0) - (buf_b_full or 0) )
        if d_buf is not None: tot_buf += d_buf
        rows.append({
            "player": pl, "sp_before": int(sp_b), "sp_after": int(sp_a), "delta_sp": int(d_sp),
            "per_account_sp_before": eff_b, "per_account_sp_after": eff_a, "per_account_sp_delta": d_by_acct,
            "best_rank_before": r_b_full, "best_rank_after": r_a_full,
            "best_rank_before_label": _rank_label(r_b_full), "best_rank_after_label": _rank_label(r_a_full),
            "buffer_before": None if buf_b_full is None else int(buf_b_full),
            "buffer_after":  None if buf_a_full is None else int(buf_a_full),
            "delta_buffer":   None if d_buf is None else int(d_buf),
            "qp_before": int(ctx_b["family_qp_player"]), "qp_after": int(ctx_a["family_qp_player"]), "delta_qp": int(d_qp)
        })

    rows.sort(key=lambda r: (-r["delta_qp"], -r["delta_sp"], r["player"].lower()))
    totals = {"delta_sp": int(sum(r["delta_sp"] for r in rows)), "delta_buffer": int(tot_buf), "delta_qp": int(fam1 - fam0)}

    # Fragility (trade-created only)
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

    frag_list = _created_fragile_firsts(det0, det1, req.defend_buffers, restrict_players=touched) if req.fragility_mode=="trade_delta" else []
    frag_note = ("No new fragile firsts created by this trade." if not frag_list else f"{len(frag_list)} new fragile first(s) created by this trade.") if req.fragility_mode=="trade_delta" else "Fragility check disabled for this request."
    verdict = "ACCEPT" if (fam1 - fam0) > 0 and len(frag_list) == 0 else ("CAUTION" if (fam1 - fam0) >= 0 else "DECLINE")

    return {
        "allocation_plan": alloc_plan,
        "per_account": {a: {"qp_before": int(per0[a]), "qp_after": int(per1[a]), "delta_qp": int(per1[a]-per0[a])} for a in FAMILY_ACCOUNTS},
        "family_qp": {"before": int(fam0), "after": int(fam1), "delta": int(fam1 - fam0)},
        "player_changes": rows, "total_changes": totals,
        "fragility_alerts": frag_list, "fragility_notes": frag_note,
        "ownership_warnings": ownership_warnings, "verdict": verdict
    }

# ---- Trade + Counter (unchanged)
@app.post("/family_trade_plus_counter_by_urls")
def family_trade_plus_counter_by_urls(req: FamilyTradePlusCounterReq):
    tags_map = _load_player_tags(req.prefer_env_defaults, req.player_tags_url)
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

    leader = normalize_leaderboard(fetch_xlsx(_pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)))
    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url, req.prefer_env_defaults, req.holdings_ud_url)

    cur, _ = _simulate_family_trade_allocation(leader, accounts, req.trade, tags_map)
    for line in [l for l in req.trade if l.side == "GIVE"]:
        players = split_multi_subject_players(line.players)
        if not players or line.sp <= 0: continue
        for p in _unique_preserve(players):
            cur[req.trade_account][p] = cur[req.trade_account].get(p, 0) - line.sp
            if cur[req.trade_account][p] <= 0:
                cur[req.trade_account].pop(p, None)

    pool_df = parse_collection_pool(req.prefer_env_defaults, req.collection_pool_url)
    trade_get_players: set[str] = {p.lower() for l in req.trade if l.side=="GET" for p in split_multi_subject_players(l.players)}
    ignore = trade_get_players if req.exclude_trade_get_players else set()

    review = _plan_collection_buys_greedy(
        leader, cur, pool_df, req.defend_buffers, req.scan_top_candidates, req.max_each, req.max_multiples_per_card,
        players_whitelist=req.players_whitelist, players_blacklist=req.players_blacklist, ignore_players=ignore,
        budget_target=req.counter_sp_target, budget_mode=req.counter_sp_mode, budget_tolerance=req.counter_sp_tolerance
    )

    return {**evaluation, "counter": review}

# ---- Collection review (greedy, unchanged)
@app.post("/family_collection_review_by_urls")
def family_collection_review_by_urls(req: FamilyCollectionReviewReq):
    tags_map = _load_player_tags(req.prefer_env_defaults, req.player_tags_url)
    leader = normalize_leaderboard(fetch_xlsx(_pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)))
    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url, req.prefer_env_defaults, req.holdings_ud_url)
    pool_df  = parse_collection_pool(req.prefer_env_defaults, req.collection_pool_url)
    review = _plan_collection_buys_greedy(
        leader, accounts, pool_df, req.defend_buffers, req.scan_top_candidates, req.max_each, req.max_multiples_per_card,
        players_whitelist=req.players_whitelist, players_blacklist=req.players_blacklist, ignore_players=None,
        tags=tags_map
    )
    return review

# ---- NEW: Collection All‑In (fast, deterministic)
@app.post("/family_collection_all_in_by_urls")
def family_collection_all_in_by_urls(req: FamilyCollectionAllInReq):
    tags_map = _load_player_tags(req.prefer_env_defaults, req.player_tags_url)
    leader   = normalize_leaderboard(fetch_xlsx(_pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)))
    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url, req.prefer_env_defaults, req.holdings_ud_url)
    pool_df  = parse_collection_pool(req.prefer_env_defaults, req.collection_pool_url)

    wl = set([(p or "").lower() for p in (req.players_whitelist or [])])
    bl = set([(p or "").lower() for p in (req.players_blacklist or [])])

    # Aggregate total add SP per player across the entire pool (respect multi‑subject)
    add_sp: Dict[str, int] = defaultdict(int)
    considered_rows = 0
    for _, row in pool_df.iterrows():
        players = split_multi_subject_players(row.get("players",""))
        if not players: continue
        if wl and not any(p.lower() in wl for p in players): continue
        if bl and any(p.lower() in bl for p in players): continue
        sp = _as_int(row.get("sp", 0)); qty = _as_int(row.get("qty", 1))
        if sp <= 0 or qty <= 0: continue
        considered_rows += 1
        for p in players:
            add_sp[p] += sp * qty

    # For display: helper to compute effective family SP map for a player
    def _eff_map(player: str, base: Dict[str, Dict[str,int]]) -> Dict[str,int]:
        return {acct: max(_lb_family_sp_for(leader, player, acct), int(base.get(acct, {}).get(player, 0))) for acct in FAMILY_ACCOUNTS}

    # Decide allocation per player (best_per_player or to_account)
    alloc_per_player: Dict[str, str] = {}
    for player, inc_sp in add_sp.items():
        if inc_sp <= 0: continue
        if req.assign_mode == "to_account" and req.assign_to_account:
            alloc_per_player[player] = req.assign_to_account
            continue

        # Evaluate each family account as the receiver of all added SP for this player
        eff_b = _eff_map(player, accounts)
        best = None; best_acct = None
        for acct in FAMILY_ACCOUNTS:
            eff_a = {**eff_b}
            eff_a[acct] = eff_a.get(acct, 0) + inc_sp

            r_b, buf_b, best_b_acct, _ = _rank_and_buffer_full_leader(player, leader, eff_b)
            r_a, buf_a, _best_a_acct, _ = _rank_and_buffer_full_leader(player, leader, eff_a)

            # Per‑player QP delta via smallset context
            ctx_b = _rank_context_smallset(player, leader, {a: {player: eff_b.get(a,0)} for a in FAMILY_ACCOUNTS})
            ctx_a = _rank_context_smallset(player, leader, {a: {player: eff_a.get(a,0)} for a in FAMILY_ACCOUNTS})
            d_qp = ctx_a["family_qp_player"] - ctx_b["family_qp_player"]

            buf_gain = 0 if (buf_b is None and buf_a is None) else max(0, (buf_a or 0) - (buf_b or 0))
            rank_sc_b = 0 if (r_b is None) else -r_b
            rank_sc_a = 0 if (r_a is None) else -r_a
            rank_gain = max(0, (rank_sc_a - rank_sc_b))
            keep_vote = 1 if (best_b_acct is not None and acct == best_b_acct) else 0

            score = (int(d_qp), int(buf_gain), int(rank_gain), int(keep_vote))
            if (best is None) or (score > best):
                best = score; best_acct = acct

        alloc_per_player[player] = best_acct or "Easystreet31"

    # Apply allocation in bulk
    accounts_after = {a: dict(v) for a, v in accounts.items()}
    for player, inc_sp in add_sp.items():
        acct = alloc_per_player.get(player)
        if acct:
            accounts_after[acct][player] = accounts_after[acct].get(player, 0) + int(inc_sp)

    fam0, per0, _ = compute_family_qp(leader, accounts)
    fam1, per1, _ = compute_family_qp(leader, accounts_after)

    # Player changes (aggregated)
    rows = []
    tot_buf = 0
    for player, inc_sp in sorted(add_sp.items(), key=lambda kv: kv[0].lower()):
        if inc_sp <= 0: continue
        eff_b = _eff_map(player, accounts)
        eff_a = _eff_map(player, accounts_after)
        sp_b = sum(eff_b.values()); sp_a = sp_b + int(inc_sp); d_sp = int(inc_sp)

        ctx_b = _rank_context_smallset(player, leader, {a: {player: eff_b.get(a,0)} for a in FAMILY_ACCOUNTS})
        ctx_a = _rank_context_smallset(player, leader, {a: {player: eff_a.get(a,0)} for a in FAMILY_ACCOUNTS})
        d_qp = ctx_a["family_qp_player"] - ctx_b["family_qp_player"]

        r_b_full, buf_b_full, _, _ = _rank_and_buffer_full_leader(player, leader, {a: eff_b.get(a,0) for a in FAMILY_ACCOUNTS})
        r_a_full, buf_a_full, _, _ = _rank_and_buffer_full_leader(player, leader, {a: eff_a.get(a,0) for a in FAMILY_ACCOUNTS})
        d_buf = None if (buf_b_full is None and buf_a_full is None) else ( (buf_a_full or 0) - (buf_b_full or 0) )
        if d_buf is not None: tot_buf += d_buf

        rows.append({
            "player": player,
            "allocated_to": alloc_per_player.get(player),
            "add_sp": int(inc_sp),
            "sp_before": int(sp_b), "sp_after": int(sp_a), "delta_sp": int(d_sp),
            "best_rank_before": r_b_full, "best_rank_after": r_a_full,
            "best_rank_before_label": _rank_label(r_b_full), "best_rank_after_label": _rank_label(r_a_full),
            "buffer_before": None if buf_b_full is None else int(buf_b_full),
            "buffer_after":  None if buf_a_full is None else int(buf_a_full),
            "delta_buffer":   None if d_buf is None else int(d_buf),
            "delta_qp": int(d_qp)
        })

    rows.sort(key=lambda r: (-r["delta_qp"], -r["delta_sp"], r["player"].lower()))
    totals = {"add_sp": int(sum(v for v in add_sp.values())), "delta_buffer": int(tot_buf), "delta_qp": int(fam1 - fam0)}

    return {
        "allocation": {"mode": req.assign_mode, "assign_to_account": req.assign_to_account, "per_player": alloc_per_player},
        "summary": {
            "family_qp": {"before": int(fam0), "after": int(fam1), "delta": int(fam1 - fam0)},
            "per_account": {a: {"qp_before": int(per0[a]), "qp_after": int(per1[a]), "delta_qp": int(per1[a]-per0[a])} for a in FAMILY_ACCOUNTS},
            "counts": {"rows_considered": int(considered_rows), "unique_players": int(len(add_sp)), "partial": False}
        },
        "player_changes": rows,
        "totals": totals
    }

# ---- Safe‑Sell report
@app.post("/family_safe_sell_report_by_urls")
def family_safe_sell_report_by_urls(req: FamilySafeSellReq):
    tags_map = _load_player_tags(req.prefer_env_defaults, req.player_tags_url)
    leader = normalize_leaderboard(fetch_xlsx(_pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)))
    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url, req.prefer_env_defaults, req.holdings_ud_url)

    wl = set([(p or "").lower() for p in (req.players_whitelist or [])])
    bl = set([(p or "").lower() for p in (req.players_blacklist or [])])
    ex = set(req.exclude_accounts or [])

    items: List[Dict[str, Any]] = []
    for acct in FAMILY_ACCOUNTS:
        if acct in ex: continue
        for player, sp_owned in sorted(accounts.get(acct, {}).items()):
            if sp_owned <= 0: continue
            p_lower = (player or "").lower()
            if wl and p_lower not in wl: continue
            if bl and p_lower in bl: continue

            rows = _smallset_entries_for_player(player, leader, accounts)
            if not rows: continue
            ordered, rank_by_key = _dedup_and_rank(rows)
            r, s = rank_by_key.get(_canon_user_strong(acct), (9999, 0))
            if not req.include_top3 and r <= 3: continue

            third_sp = ordered[2][2] if len(ordered) > 2 else (ordered[1][2] if len(ordered) > 1 else 0)
            distance_to_rank3 = max((third_sp + 1) - s, 0)
            if distance_to_rank3 < int(req.min_distance_to_rank3): continue

            items.append({
                "account": acct, "player": player, "sp_owned": int(s), "rank": int(r),
                "third_sp": int(third_sp), "distance_to_rank3": int(distance_to_rank3)
            })

    items.sort(key=lambda d: (-d["distance_to_rank3"], -d["sp_owned"], d["player"].lower()))
    per_acct: Dict[str, List[Dict[str, Any]]] = {a: [] for a in FAMILY_ACCOUNTS}
    for it in items: per_acct[it["account"]].append(it)

    return {
        "per_account": per_acct,
        "flat": items,
        "summary": {
            "filters": {
                "include_top3": bool(req.include_top3),
                "min_distance_to_rank3": int(req.min_distance_to_rank3),
                "whitelist_count": len(wl) if wl else 0,
                "blacklist_count": len(bl) if bl else 0,
                "excluded_accounts": list(ex) if ex else []
            },
            "counts": {"returned": len(items), "accounts_with_items": sum(1 for a in FAMILY_ACCOUNTS if per_acct[a])}
        }
    }

# ---- Leaderboard delta
@app.post("/leaderboard_delta_by_urls")
def leaderboard_delta_by_urls(req: LeaderboardDeltaReq):
    def _to_maps(lst: List[Dict[str, Any]]):
        key_to_sp: Dict[str, int] = {}; key_to_label: Dict[str, str] = {}
        for e in lst:
            raw_user = str(e["user"]); key = _canon_user_strong(raw_user); sp = _as_int(e["sp"])
            if key not in key_to_sp or sp > key_to_sp[key]:
                key_to_sp[key] = sp; key_to_label.setdefault(key, raw_user)
        return key_to_sp, key_to_label

    today = normalize_leaderboard(fetch_xlsx(_sanitize_gsheet_url(req.leaderboard_today_url)))
    yday  = normalize_leaderboard(fetch_xlsx(_sanitize_gsheet_url(req.leaderboard_yesterday_url)))

    changes = []
    players = sorted(set(list(today.keys()) + list(yday.keys())))
    for p in players:
        tm, tlabels = _to_maps(today.get(p, [])); ym, ylabels = _to_maps(yday.get(p, []))
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

# ---- Fragile whitelist (1sts & 2nds)
@app.post("/family_fragile_whitelist_by_urls")
def family_fragile_whitelist_by_urls(req: FamilyFragileWhitelistReq):
    leader = normalize_leaderboard(fetch_xlsx(_pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)))
    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url, req.prefer_env_defaults, req.holdings_ud_url)

    fragile_firsts: List[Dict[str, Any]] = []
    fragile_seconds: List[Dict[str, Any]] = []

    all_players = set(leader.keys())
    for a in FAMILY_ACCOUNTS:
        all_players.update(accounts.get(a, {}).keys())

    for player in sorted(all_players):
        rows = _smallset_entries_for_player(player, leader, accounts)
        if not rows: continue
        ordered, rank_by_key = _dedup_and_rank(rows)
        for acct in FAMILY_ACCOUNTS:
            key = _canon_user_strong(acct)
            rank, sp = rank_by_key.get(key, (9999, 0))
            if rank not in (1, 2): continue
            target = int(req.defend_buffers.get(acct, DEFAULT_DEFEND_BUFFER))
            if rank == 1 and req.include_firsts:
                second_sp = ordered[1][2] if len(ordered) > 1 else 0
                cushion = int(sp - second_sp); need = max(target - cushion, 0)
                if cushion < target:
                    nearest_threat = ordered[1][1] if len(ordered) > 1 else None
                    fragile_firsts.append({
                        "player": player, "account": acct, "cushion": cushion, "target": target,
                        "need_sp_for_target": need, "current_sp": int(sp), "second_sp": int(second_sp),
                        "nearest_threat": nearest_threat
                    })
            if rank == 2 and req.include_seconds:
                third_sp = ordered[2][2] if len(ordered) > 2 else 0
                cushion = int(sp - third_sp); need = max(target - cushion, 0)
                if cushion < target:
                    nearest_threat = ordered[2][1] if len(ordered) > 2 else None
                    gap_to_first = int(max(ordered[0][2] - sp, 0))
                    fragile_seconds.append({
                        "player": player, "account": acct, "cushion": cushion, "target": target,
                        "need_sp_for_target": need, "current_sp": int(sp), "third_sp": int(third_sp),
                        "gap_to_first": gap_to_first, "nearest_threat": nearest_threat
                    })

    fragile_firsts.sort(key=lambda r: (r["cushion"], -r["need_sp_for_target"], r["player"].lower()))
    fragile_seconds.sort(key=lambda r: (r["cushion"], -r["need_sp_for_target"], r["player"].lower()))

    names = []; seen = set()
    for src in (fragile_firsts, fragile_seconds):
        for r in src:
            p = r["player"]
            if p.lower() not in seen:
                seen.add(p.lower()); names.append(p)
            if len(names) >= max(0, int(req.limit)): break
        if len(names) >= max(0, int(req.limit)): break

    return {
        "whitelist": names,
        "fragile_firsts": fragile_firsts,
        "fragile_seconds": fragile_seconds,
        "summary": {
            "counts": {"fragile_firsts": len(fragile_firsts), "fragile_seconds": len(fragile_seconds), "unique_players": len(names)},
            "limit": int(req.limit),
            "defend_buffers": req.defend_buffers
        }
    }
