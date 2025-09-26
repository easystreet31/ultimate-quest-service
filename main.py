# main.py  (v4.0.4-phase23-defaults)
import io, os, re, math, json
from typing import List, Dict, Any, Optional, Tuple, Set, Literal
from collections import defaultdict, Counter

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except Exception:
    HAS_ORTOOLS = False

APP_VERSION = "4.0.4-phase23-defaults"

# -------------------------------------------------------------------
# Built-in DEFAULT URLS (overridable by request payloads or env vars)
# -------------------------------------------------------------------
DEFAULT_LINKS = {
    # Leaderboard (Top-5 per player workbook)
    "leaderboard": os.getenv("DEFAULT_LEADERBOARD_URL",
        "https://docs.google.com/spreadsheets/d/1aRBznhcgaw8wCYaYAsdC14JxC3DwJJzB/export?format=xlsx"),
    # 1 day lag leaderboard
    "1 day lag leaderboard": os.getenv("DEFAULT_LEADERBOARD_YDAY_URL",
        "https://docs.google.com/spreadsheets/d/1MRpJWiO9qBPuz5CGfLXgS8t4DWooyeN2/export?format=xlsx"),
    # Primary account
    "holdings_e31": os.getenv("DEFAULT_HOLDINGS_E31_URL",
        "https://docs.google.com/spreadsheets/d/1J3jf8c5M0oYbhStLLOS-ItRSAvOTvfkS5ggp0yl5XgY/export?format=xlsx"),
    "collection_e31": os.getenv("DEFAULT_COLLECTION_E31_URL",
        "https://docs.google.com/spreadsheets/d/1C6sW2VTE0ezuPNxYv4l-lTJ_wGl1HE-7wBveTslVrUs/export?format=xlsx"),
    # Alt accounts
    "holdings_dc": os.getenv("DEFAULT_HOLDINGS_DC_URL",
        "https://docs.google.com/spreadsheets/d/1OTGjLHXzMeYP7H0bzWtLQZ-3qRF1DXRSIvDzHVUcROM/export?format=xlsx"),
    "collection_dc": os.getenv("DEFAULT_COLLECTION_DC_URL",
        "https://docs.google.com/spreadsheets/d/1_hx66ZDPOO6a2RVOOLbuaxqPknBomdDkkX2S4WLj7yg/export?format=xlsx"),
    "holdings_fe": os.getenv("DEFAULT_HOLDINGS_FE_URL",
        "https://docs.google.com/spreadsheets/d/1qCvE5lN4LJsaG_bsCWaxJ1cYVSc3eSR7D2UKA6dLdHs/export?format=xlsx"),
    "collection_fe": os.getenv("DEFAULT_COLLECTION_FE_URL",
        "https://docs.google.com/spreadsheets/d/1YAGAiyy9V7NdoUP0M2885pxOh_asWn3JPl9sjvjfqFk/export?format=xlsx"),
    # Seller/pool collection
    "pool_collection": os.getenv("DEFAULT_POOL_COLLECTION_URL",
        "https://docs.google.com/spreadsheets/d/1tto_9aow568dSm-3Wqk-S-YaCh7tQg_UmFJEyIjL6DQ/export?format=xlsx"),

}

DEFAULT_RIVALS = [
    "chfkyle", "Tfunite", "FireRanger", "VJV5", "Erikk", "tommyknockrs76"
]

FAMILY_ACCOUNTS = ["Easystreet31", "DusterCrusher", "FinkleIsEinhorn"]

app = FastAPI(title="Ultimate Quest Service", version=APP_VERSION)

# ---------------------------
# Utilities
# ---------------------------

def _default(url: Optional[str], key: str) -> str:
    return url or DEFAULT_LINKS[key]

def fetch_xlsx(url: str) -> Dict[str, pd.DataFrame]:
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = io.BytesIO(resp.content)
        xls = pd.ExcelFile(data)
        sheets = {}
        for name in xls.sheet_names:
            df = xls.parse(name)
            df.columns = [str(c).strip() for c in df.columns]
            sheets[name] = df
        return sheets
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read XLSX from URL: {url} ({e})")

def _norm_player(p: Any) -> str:
    if pd.isna(p): return ""
    return re.sub(r"\s+", " ", str(p).strip())

def _norm_user(u: Any) -> str:
    if pd.isna(u): return ""
    return re.sub(r"\s+", " ", str(u).strip())

def _canon_user(u: Any) -> str:
    s = _norm_user(u)
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)  # strip trailing " (1234)"
    return s.strip()

def _as_int(x: Any) -> int:
    try:
        if pd.isna(x) or x is None or x == "": return 0
        if isinstance(x, (int, np.integer)): return int(x)
        if isinstance(x, float) and math.isfinite(x): return int(round(x))
        s = str(x).strip().replace(",", "")
        if s == "": return 0
        return int(round(float(s)))
    except Exception:
        return 0

def _safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj

def split_multi_subject_players(players_text: str) -> List[str]:
    if not players_text:
        return []
    parts = re.split(r"\s*(?:/|&|\+|,| and |\||—|–)\s*", str(players_text))
    parts = [_norm_player(p) for p in parts if _norm_player(p)]
    unique, seen = [], set()
    for p in parts:
        if p.lower() not in seen:
            seen.add(p.lower())
            unique.append(p)
    return unique

def qp_for_rank(rank: int) -> int:
    return 5 if rank == 1 else 3 if rank == 2 else 1 if rank == 3 else 0

def normalize_leaderboard(sheets: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
    candidate = None
    for name, df in sheets.items():
        if any("player" in c.lower() for c in df.columns):
            candidate = df
            break
    if candidate is None:
        candidate = list(sheets.values())[0]
    df = candidate.copy()
    df.columns = [c.strip() for c in df.columns]

    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    lower = [c.lower() for c in df.columns]
    has_player = any("player" in c for c in lower)
    has_user   = any(("user" in c or "owner" in c or "name" in c) for c in lower)
    has_sp     = any(("sp" in c or "points" in c or "subject" in c) for c in lower)

    if has_player and has_user and has_sp:
        player_col = next(c for c in df.columns if "player" in c.lower())
        user_col   = next(c for c in df.columns if any(k in c.lower() for k in ["user","owner","name"]))
        sp_col     = next(c for c in df.columns if any(k in c.lower() for k in ["sp","points","subject"]))
        for _, row in df.iterrows():
            p = _norm_player(row.get(player_col, ""))
            u = _canon_user(row.get(user_col, ""))
            sp = _as_int(row.get(sp_col, 0))
            if p and u:
                out[p].append(dict(user=u, sp=sp))
        for p in list(out.keys()):
            best = {}
            for e in out[p]:
                u = e["user"]
                if u not in best or e["sp"] > best[u]:
                    best[u] = e["sp"]
            merged = [dict(user=u, sp=best[u]) for u in best]
            merged.sort(key=lambda x: (-x["sp"], x["user"].lower()))
            out[p] = merged
        return out

    # Wide Top-N layout
    player_col = next((c for c in df.columns if "player" in c.lower()), df.columns[0])
    pairs = []
    cols = df.columns.tolist()
    for c in cols:
        if any(k in c.lower() for k in ["user","owner","name"]):
            spcol = None
            idx = cols.index(c)
            if idx + 1 < len(cols):
                if any(k in cols[idx+1].lower() for k in ["sp","points"]):
                    spcol = cols[idx+1]
            if spcol is None:
                for c2 in cols:
                    if any(k in c2.lower() for k in ["sp","points"]):
                        if any(d.isdigit() for d in c) and any(d.isdigit() for d in c2):
                            if "".join(d for d in c if d.isdigit()) == "".join(d for d in c2 if d.isdigit()):
                                spcol = c2; break
                if spcol is None:
                    for c2 in cols:
                        if any(k in c2.lower() for k in ["sp","points"]) and c2 != player_col:
                            spcol = c2; break
            pairs.append((c, spcol))
    for _, row in df.iterrows():
        p = _norm_player(row.get(player_col, ""))
        if not p: continue
        bucket = []
        for name_col, sp_col in pairs:
            u = _canon_user(row.get(name_col, ""))
            sp = _as_int(row.get(sp_col, 0) if sp_col else 0)
            if u:
                bucket.append(dict(user=u, sp=sp))
        if bucket:
            bucket.sort(key=lambda x: (-x["sp"], x["user"].lower()))
            out[p].extend(bucket)
    for p, lst in out.items():
        best = {}
        for e in lst:
            u = e["user"]
            if u not in best or e["sp"] > best[u]:
                best[u] = e["sp"]
        merged = [dict(user=u, sp=best[u]) for u in best]
        merged.sort(key=lambda x: (-x["sp"], x["user"].lower()))
        out[p] = merged
    return out

def parse_holdings(sheets: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    df = None
    for _, d in sheets.items():
        if len(d.columns) > 0:
            df = d; break
    if df is None:
        return {}
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    pcol = next((c for c in df.columns if any(k in c.lower() for k in ["player","players"])), df.columns[0])
    scol = next((c for c in df.columns if any(k in c.lower() for k in ["sp","subject"])), None)
    if scol is None:
        scol = next((c for c in df.columns if "total" in c.lower() and "sp" in c.lower()), df.columns[-1])
    totals: Dict[str, int] = defaultdict(int)
    for _, row in df.iterrows():
        plist = split_multi_subject_players(row.get(pcol, ""))
        sp = _as_int(row.get(scol, 0))
        if plist:
            for p in plist:
                totals[p] += sp
    return dict(totals)

def parse_collection(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = None
    for _, d in sheets.items():
        if len(d.index) > 0:
            df = d; break
    if df is None:
        return pd.DataFrame(columns=["card","no","players","sp","qty"])
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    col_card = next((c for c in df.columns if any(k in c.lower() for k in ["card", "title", "name"])), df.columns[0])
    col_no   = next((c for c in df.columns if any(k in c.lower() for k in ["no", "number", "card #"])), None)
    col_pl   = next((c for c in df.columns if any(k in c.lower() for k in ["player", "players", "subject"])), None)
    col_sp   = next((c for c in df.columns if "sp" in c.lower() or "subject" in c.lower()), None)
    col_qty  = next((c for c in df.columns if any(k in c.lower() for k in ["qty","quantity","count"])), None)
    out = pd.DataFrame({
        "card": df.get(col_card, ""),
        "no": df.get(col_no, ""),
        "players": df.get(col_pl, ""),
        "sp": df.get(col_sp, 0).apply(_as_int),
        "qty": df.get(col_qty, 1).apply(_as_int),
    })
    out["players"] = out["players"].fillna("").astype(str)
    out["card"] = out["card"].fillna("").astype(str)
    out["no"] = out["no"].fillna("").astype(str)
    out["sp"] = out["sp"].fillna(0).astype(int)
    out["qty"] = out["qty"].fillna(1).astype(int)
    out = out[out["card"].str.strip() != ""].copy()
    out.reset_index(drop=True, inplace=True)
    return out

def compute_rank_and_margins(leader: Dict[str, List[Dict[str, Any]]],
                             accounts_sp: Dict[str, Dict[str, int]],
                             player: str) -> Dict[str, Dict[str, Any]]:
    comp_entries = [(e["user"], e["sp"]) for e in leader.get(player, []) if e["user"] not in FAMILY_ACCOUNTS]
    comp_max_sp = max([s for _, s in comp_entries], default=0)
    rows = []
    for (u, s) in comp_entries:
        rows.append(("__competitor__", u, s))
    for acct in FAMILY_ACCOUNTS:
        s = accounts_sp.get(acct, {}).get(player, 0)
        rows.append(("__family__", acct, s))
    rows.sort(key=lambda x: (-x[2], x[1].lower()))
    ranks: Dict[str, int] = {}
    ordered = []
    last_sp = None
    current_rank = 0
    seen_count = 0
    for typ, name, sp in rows:
        seen_count += 1
        if last_sp is None or sp < last_sp:
            current_rank = seen_count
        last_sp = sp
        ordered.append((typ, name, sp, current_rank))
        if typ == "__family__":
            ranks[name] = current_rank
    first_sp = ordered[0][2] if ordered else 0
    out: Dict[str, Dict[str, Any]] = {}
    for acct in FAMILY_ACCOUNTS:
        s = accounts_sp.get(acct, {}).get(player, 0)
        r = ranks.get(acct, None)
        qp = qp_for_rank(r) if r else 0
        margin_if_first = s - comp_max_sp if r == 1 else None
        gap_to_first = first_sp - s if r and r > 1 else (first_sp - s if s < first_sp else 0)
        out[acct] = dict(sp=s, rank=r or 9999, qp=qp, margin_if_first=margin_if_first, gap_to_first=gap_to_first)
    return out

def compute_family_qp(leader: Dict[str, List[Dict[str, Any]]],
                      accounts_sp: Dict[str, Dict[str, int]]) -> Tuple[int, Dict[str, int], Dict[str, Dict[str, Any]]]:
    per_account_qp: Dict[str, int] = {a: 0 for a in FAMILY_ACCOUNTS}
    per_player_details: Dict[str, Dict[str, Dict[str, Any]]] = {a: {} for a in FAMILY_ACCOUNTS}
    all_players: Set[str] = set(leader.keys())
    for a in FAMILY_ACCOUNTS:
        all_players.update(accounts_sp.get(a, {}).keys())
    for player in all_players:
        stats = compute_rank_and_margins(leader, accounts_sp, player)
        for acct in FAMILY_ACCOUNTS:
            per_player_details[acct][player] = stats[acct]
            per_account_qp[acct] += stats[acct]["qp"]
    fam_qp = sum(per_account_qp.values())
    return fam_qp, per_account_qp, per_player_details

def apply_card_sp_to_account(acct_sp: Dict[str, int], players: List[str], sp: int, add: bool, rule: str = "full_each_unique"):
    unique = list({p: True for p in players}.keys())
    if not unique: return
    if rule == "full_each_unique":
        for p in unique:
            acct_sp[p] = acct_sp.get(p, 0) + (sp if add else -sp)
    else:
        for p in unique:
            acct_sp[p] = acct_sp.get(p, 0) + (sp if add else -sp)

def clone_accounts_sp(accounts_sp: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    return {a: dict(v) for a, v in accounts_sp.items()}

def fragile_firsts(per_player_details: Dict[str, Dict[str, Any]], buffer_val: int) -> List[str]:
    frag = []
    for player, d in per_player_details.items():
        if d.get("rank", 9999) == 1:
            margin = d.get("margin_if_first", None)
            if margin is not None and margin < buffer_val:
                frag.append(f"{player} (margin {margin})")
    return sorted(frag)

def holdings_from_urls(e31_url: Optional[str], dc_url: Optional[str], fe_url: Optional[str]) -> Dict[str, Dict[str, int]]:
    out = {a: {} for a in FAMILY_ACCOUNTS}
    out["Easystreet31"] = parse_holdings(fetch_xlsx(_default(e31_url, "holdings_e31")))
    out["DusterCrusher"] = parse_holdings(fetch_xlsx(_default(dc_url, "holdings_dc")))
    out["FinkleIsEinhorn"] = parse_holdings(fetch_xlsx(_default(fe_url, "holdings_fe")))
    return out

def apply_trade_lines_to_accounts(accounts_sp: Dict[str, Dict[str, int]],
                                  trade: List["TradeLine"],
                                  rule: str,
                                  leader: Dict[str, List[Dict[str, Any]]],
                                  allocate_get_to_best: bool,
                                  trade_account: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    alloc_plan = []
    cur = clone_accounts_sp(accounts_sp)
    fam_qp0, _, _ = compute_family_qp(leader, cur)
    for line in [l for l in trade if l.side == "GET"]:
        players = split_multi_subject_players(line.players)
        best_acct = None; best_gain = -10**9; best_snapshot = None
        for acct in FAMILY_ACCOUNTS:
            sim = clone_accounts_sp(cur)
            for p in players:
                apply_card_sp_to_account(sim[acct], [p], line.sp, add=True, rule=rule)
            fam_qp, _, _ = compute_family_qp(leader, sim)
            gain = fam_qp - fam_qp0
            if gain > best_gain:
                best_gain = gain; best_acct = acct; best_snapshot = sim
        cur = best_snapshot if best_snapshot else cur
        fam_qp0, _, _ = compute_family_qp(leader, cur)
        alloc_plan.append(dict(type="GET", players=players, sp=line.sp, to=best_acct, family_qp_gain=best_gain))
    for line in [l for l in trade if l.side == "GIVE"]:
        players = split_multi_subject_players(line.players)
        for p in players:
            apply_card_sp_to_account(cur[trade_account], [p], line.sp, add=False, rule=rule)
        alloc_plan.append(dict(type="GIVE", players=players, sp=line.sp, from_acct=trade_account))
    return alloc_plan, cur

# ---------------------------
# Pydantic models (URLS optional now)
# ---------------------------

class TradeLine(BaseModel):
    side: Literal["GET","GIVE"] = Field(description="GET or GIVE")
    players: str
    sp: int

class EvaluateReqSingle(BaseModel):
    leaderboard_url: Optional[str] = None
    holdings_url: Optional[str] = None
    multi_subject_rule: str = "full_each_unique"
    defend_buffer: int = 20
    scope: str = "trade_only"
    max_return_players: int = 120
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None
    trade: List[TradeLine]

class ScanReq(BaseModel):
    leaderboard_url: Optional[str] = None
    holdings_url: Optional[str] = None
    defend_buffer: int = 20
    upgrade_gap: int = 12
    entry_gap: int = 8
    keep_buffer: int = 30
    max_each: int = 25
    show_all: bool = False
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None

class RivalScanReq(BaseModel):
    leaderboard_url: Optional[str] = None
    holdings_url: Optional[str] = None
    focus_rival: str
    defend_buffer: int = 20
    upgrade_gap: int = 12
    entry_gap: int = 8
    keep_buffer: int = 30
    max_each: int = 25
    show_all: bool = False

class PartnerScanReq(BaseModel):
    leaderboard_url: Optional[str] = None
    holdings_url: Optional[str] = None
    partner: str
    target_rivals: Optional[List[str]] = None
    protect_qp: bool = True
    protect_buffer: int = 20
    max_sp_to_gain: int = 25
    max_each: int = 50
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None

class CollectionReviewReq(BaseModel):
    leaderboard_url: Optional[str] = None
    holdings_url: Optional[str] = None
    collection_url: Optional[str] = None
    multi_subject_rule: str = "full_each_unique"
    defend_buffer: int = 20
    focus_rival: Optional[str] = None
    rival_only: bool = False
    max_each: int = 60
    max_multiples_per_card: int = 3
    scan_top_candidates: int = 60
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None
    baseline_trade: Optional[List[TradeLine]] = None

class SafeGiveReq(BaseModel):
    leaderboard_url: Optional[str] = None
    holdings_url: Optional[str] = None
    my_collection_url: Optional[str] = None
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

class FamilyEvaluateTradeReq(BaseModel):
    leaderboard_url: Optional[str] = None
    holdings_e31_url: Optional[str] = None
    holdings_dc_url: Optional[str] = None
    holdings_fe_url: Optional[str] = None
    trade: List[TradeLine]
    trade_account: str = "Easystreet31"
    multi_subject_rule: str = "full_each_unique"
    defend_buffers: Dict[str, int] = Field(default_factory=lambda: {
        "Easystreet31": 15, "DusterCrusher": 15, "FinkleIsEinhorn": 15
    })
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None

class CollectionReviewFamilyReq(BaseModel):
    leaderboard_url: Optional[str] = None
    holdings_e31_url: Optional[str] = None
    holdings_dc_url: Optional[str] = None
    holdings_fe_url: Optional[str] = None
    collection_url: Optional[str] = None
    multi_subject_rule: str = "full_each_unique"
    defend_buffer: int = 15
    rival_only: bool = False
    focus_rival: Optional[str] = None
    max_each: int = 60
    max_multiples_per_card: int = 3
    scan_top_candidates: int = 60
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None
    baseline_trade: Optional[List[TradeLine]] = None

class LeaderboardDeltaReq(BaseModel):
    leaderboard_today_url: Optional[str] = None
    leaderboard_yesterday_url: Optional[str] = None  # if omitted, must be provided via env or call
    me_accounts: List[str] = Field(default_factory=lambda: FAMILY_ACCOUNTS.copy())
    rivals: Optional[List[str]] = None
    show_only_changes: bool = True
    min_sp_delta: int = 1
    min_rank_change: int = 1

class FamilyTransferReq(BaseModel):
    leaderboard_url: Optional[str] = None
    holdings_e31_url: Optional[str] = None
    holdings_dc_url: Optional[str] = None
    holdings_fe_url: Optional[str] = None
    collection_e31_url: Optional[str] = None
    collection_dc_url: Optional[str] = None
    collection_fe_url: Optional[str] = None
    defend_buffer_all: int = 15
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None
    max_each: int = 100
    scan_top_candidates: int = 60
    max_multiples_per_card: int = 3
    target_rivals: Optional[List[str]] = None
    return_sections: Optional[List[str]] = None
    assignment_account_filter: str = "all"
    min_assignment_qp: int = 0
    limit: int = 100
    offset: int = 0

# ---------------------------
# Routes
# ---------------------------

@app.get("/info")
def info():
    return dict(
        version=APP_VERSION,
        routes=[r.path for r in app.routes],
        defaults=dict(DEFAULT_LINKS),
        default_rivals=DEFAULT_RIVALS
    )

@app.get("/health")
def health():
    return dict(ok=True)

@app.get("/config")
def config():
    return dict(DEFAULT_LINKS=DEFAULT_LINKS, DEFAULT_RIVALS=DEFAULT_RIVALS)

@app.post("/evaluate_by_urls_easystreet31")
def evaluate_by_urls_easystreet31(req: EvaluateReqSingle):
    leader = normalize_leaderboard(fetch_xlsx(_default(req.leaderboard_url, "leaderboard")))
    acct = parse_holdings(fetch_xlsx(_default(req.holdings_url, "holdings_e31")))
    accounts = {a: {} for a in FAMILY_ACCOUNTS}
    accounts["Easystreet31"] = acct
    alloc, cur = apply_trade_lines_to_accounts(accounts, req.trade, req.multi_subject_rule, leader,
                                               allocate_get_to_best=False, trade_account="Easystreet31")
    fam_before, per_qp_before, _ = compute_family_qp(leader, accounts)
    fam_after, per_qp_after, det_after = compute_family_qp(leader, cur)
    return _safe_json({
        "params": req.model_dump(),
        "allocation": alloc,
        "qp_summary": {
            "Easystreet31_before": per_qp_before["Easystreet31"],
            "Easystreet31_after": per_qp_after["Easystreet31"],
            "family_delta_qp": fam_after - fam_before
        },
        "fragile_after": fragile_firsts(det_after["Easystreet31"], req.defend_buffer),
        "verdict": "ACCEPT" if (fam_after - fam_before) > 0 else "DECLINE"
    })

@app.post("/scan_by_urls_easystreet31")
def scan_by_urls_easystreet31(req: ScanReq):
    leader = normalize_leaderboard(fetch_xlsx(_default(req.leaderboard_url, "leaderboard")))
    acct = parse_holdings(fetch_xlsx(_default(req.holdings_url, "holdings_e31")))
    accounts = {a: {} for a in FAMILY_ACCOUNTS}
    accounts["Easystreet31"] = acct
    fam_qp, per_qp, details = compute_family_qp(leader, accounts)
    thin, upgrades, entries, overs = [], [], [], []
    for player, d in details["Easystreet31"].items():
        rank = d["rank"]; sp = d["sp"]
        if rank == 1 and (d["margin_if_first"] or 0) <= req.defend_buffer:
            thin.append(dict(player=player, margin=d["margin_if_first"], sp=sp))
        if rank > 1 and d["gap_to_first"] <= req.upgrade_gap:
            upgrades.append(dict(player=player, need=d["gap_to_first"], sp=sp))
        if rank > 3 and d["gap_to_first"] <= req.entry_gap:
            entries.append(dict(player=player, gap_to_first=d["gap_to_first"]))
        if rank == 1:
            slack = (d["margin_if_first"] or 0) - req.keep_buffer
            if slack > 0:
                overs.append(dict(player=player, slack=slack))
    def cap(lst): 
        if req.show_all or req.max_each <= 0: return lst
        return lst[:req.max_each]
    return _safe_json({
        "thin_firsts": cap(sorted(thin, key=lambda x: (x["margin"] or 0))),
        "upgrade_opps": cap(sorted(upgrades, key=lambda x: x["need"])),
        "top3_entries": cap(sorted(entries, key=lambda x: x["gap_to_first"])),
        "overshoots": cap(sorted(overs, key=lambda x: -x["slack"]))
    })

@app.post("/scan_rival_by_urls_easystreet31")
def scan_rival_by_urls_easystreet31(req: RivalScanReq):
    leader = normalize_leaderboard(fetch_xlsx(_default(req.leaderboard_url, "leaderboard")))
    acct = parse_holdings(fetch_xlsx(_default(req.holdings_url, "holdings_e31")))
    accounts = {a: {} for a in FAMILY_ACCOUNTS}
    accounts["Easystreet31"] = acct
    _, _, details = compute_family_qp(leader, accounts)
    focus = _canon_user(req.focus_rival)
    threats = []
    for player, lst in leader.items():
        rival = next((e for e in lst if _canon_user(e["user"]).lower() == focus.lower()), None)
        you = details["Easystreet31"].get(player, {"sp": 0, "rank": 9999})
        if rival:
            margin_vs_rival = you.get("sp", 0) - rival["sp"]
            if you.get("rank", 999) == 1 and margin_vs_rival <= req.defend_buffer:
                threats.append(dict(player=player, your_sp=you.get("sp", 0), rival_sp=rival["sp"], margin=margin_vs_rival))
    threats.sort(key=lambda x: x["margin"])
    return _safe_json({"rival": focus, "threats": threats})

@app.post("/scan_partner_by_urls_easystreet31")
def scan_partner_by_urls_easystreet31(req: PartnerScanReq):
    leader = normalize_leaderboard(fetch_xlsx(_default(req.leaderboard_url, "leaderboard")))
    acct = parse_holdings(fetch_xlsx(_default(req.holdings_url, "holdings_e31")))
    accounts = {a: {} for a in FAMILY_ACCOUNTS}
    accounts["Easystreet31"] = acct
    _, _, details0 = compute_family_qp(leader, accounts)
    partner = _canon_user(req.partner)
    suggestions = []
    for player, lst in leader.items():
        you = details0["Easystreet31"].get(player, {"rank": 999, "sp": 0})
        part = next((e for e in lst if _canon_user(e["user"]).lower() == partner.lower()), None)
        psp = part["sp"] if part else 0
        first_sp = lst[0]["sp"] if lst else 0
        need_to_first = max(0, first_sp - psp + 1)
        if need_to_first <= req.max_sp_to_gain:
            if you.get("rank", 999) == 1:
                margin = you.get("margin_if_first", 0)
                if margin - need_to_first < req.protect_buffer:
                    continue
            suggestions.append(dict(player=player, partner_current_sp=psp, add_sp=need_to_first))
    suggestions.sort(key=lambda x: x["add_sp"])
    if req.max_each > 0:
        suggestions = suggestions[:req.max_each]
    return _safe_json({"partner": partner, "suggestions": suggestions})

@app.post("/review_collection_by_urls_easystreet31")
def review_collection_by_urls_easystreet31(req: CollectionReviewReq):
    leader = normalize_leaderboard(fetch_xlsx(_default(req.leaderboard_url, "leaderboard")))
    acct = parse_holdings(fetch_xlsx(_default(req.holdings_url, "holdings_e31")))
    col = parse_collection(fetch_xlsx(_default(req.collection_url, "pool_collection")))
    if req.baseline_trade:
        tmp_accounts = {a: {} for a in FAMILY_ACCOUNTS}
        tmp_accounts["Easystreet31"] = dict(acct)
        _, tmp_accounts = apply_trade_lines_to_accounts(tmp_accounts, req.baseline_trade, req.multi_subject_rule, leader, False, "Easystreet31")
        acct = tmp_accounts["Easystreet31"]
    accounts = {a: {} for a in FAMILY_ACCOUNTS}
    accounts["Easystreet31"] = acct
    fam0, _, _ = compute_family_qp(leader, accounts)
    picks = []
    subset = col.head(req.scan_top_candidates) if req.scan_top_candidates > 0 else col
    for _, row in subset.iterrows():
        players = split_multi_subject_players(row["players"])
        sp = int(row["sp"]); qty = int(row["qty"])
        if not players or sp <= 0 or qty <= 0:
            continue
        take = min(qty, req.max_multiples_per_card)
        sim_accounts = {a: dict(v) for a, v in accounts.items()}
        for t in range(1, take+1):
            apply_card_sp_to_account(sim_accounts["Easystreet31"], players, sp, add=True, rule=req.multi_subject_rule)
            fam, _, _ = compute_family_qp(leader, sim_accounts)
            delta = fam - fam0
            if delta > 0:
                picks.append(dict(card=row["card"], no=row["no"], players=players, take_n=t, sp=sp, family_delta_qp=delta))
            else:
                break
    picks.sort(key=lambda x: (-x["family_delta_qp"], x["card"]))
    if req.max_each > 0:
        picks = picks[:req.max_each]
    return _safe_json({"picks": picks, "count": len(picks)})

@app.post("/suggest_give_from_collection_by_urls_easystreet31")
def suggest_give_from_collection_by_urls_easystreet31(req: SafeGiveReq):
    leader = normalize_leaderboard(fetch_xlsx(_default(req.leaderboard_url, "leaderboard")))
    acct = parse_holdings(fetch_xlsx(_default(req.holdings_url, "holdings_e31")))
    mycol = parse_collection(fetch_xlsx(_default(req.my_collection_url, "collection_e31")))
    accounts = {a: {} for a in FAMILY_ACCOUNTS}
    accounts["Easystreet31"] = acct
    fam0, per0, _ = compute_family_qp(leader, accounts)
    suggestions = []
    subset = mycol.head(2000)
    for _, row in subset.iterrows():
        players = split_multi_subject_players(row["players"])
        sp = int(row["sp"]); qty = int(row["qty"])
        if not players or sp <= 0 or qty <= 0:
            continue
        sim = {a: dict(v) for a, v in accounts.items()}
        apply_card_sp_to_account(sim["Easystreet31"], players, sp, add=False, rule=req.multi_subject_rule)
        fam_after, per_after, det_after = compute_family_qp(leader, sim)
        if req.protect_qp and per_after["Easystreet31"] < per0["Easystreet31"]:
            continue
        frag_after = fragile_firsts(det_after["Easystreet31"], req.protect_buffer)
        if frag_after:
            continue
        suggestions.append(dict(card=row["card"], no=row["no"], players=players, give_n=1, sp=sp))
    suggestions = suggestions[:req.max_each] if req.max_each > 0 else suggestions
    return _safe_json({"partner": _canon_user(req.partner), "safe_give": suggestions})

# ---------------------------
# Phase 2/3 family endpoints
# ---------------------------

@app.post("/family_evaluate_trade_by_urls")
def family_evaluate_trade_by_urls(req: FamilyEvaluateTradeReq):
    leader = normalize_leaderboard(fetch_xlsx(_default(req.leaderboard_url, "leaderboard")))
    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url)
    fam0, per0, _ = compute_family_qp(leader, accounts)
    alloc, after_accounts = apply_trade_lines_to_accounts(accounts, req.trade, req.multi_subject_rule, leader,
                                                          allocate_get_to_best=True, trade_account=req.trade_account)
    fam1, per1, det1 = compute_family_qp(leader, after_accounts)
    frag = {a: fragile_firsts(det1[a], req.defend_buffers.get(a, 15)) for a in FAMILY_ACCOUNTS}
    decision = "ACCEPT" if (fam1 - fam0) > 0 and all(len(f)==0 for f in frag.values()) else (
               "CAUTION" if (fam1 - fam0) >= 0 else "DECLINE")
    return _safe_json({
        "allocation_plan": alloc,
        "per_account": {
            a: {"qp_before": per0[a], "qp_after": per1[a], "delta_qp": per1[a]-per0[a], "fragile_after": frag[a]}
            for a in FAMILY_ACCOUNTS
        },
        "family_qp": {"before": fam0, "after": fam1, "delta": fam1 - fam0},
        "verdict": decision
    })

@app.post("/collection_review_family_by_urls")
def collection_review_family_by_urls(req: CollectionReviewFamilyReq):
    leader = normalize_leaderboard(fetch_xlsx(_default(req.leaderboard_url, "leaderboard")))
    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url)
    fam0, _, _ = compute_family_qp(leader, accounts)
    col = parse_collection(fetch_xlsx(_default(req.collection_url, "pool_collection")))
    subset = col.head(req.scan_top_candidates) if req.scan_top_candidates > 0 else col
    suggestions = []
    for _, row in subset.iterrows():
        players = split_multi_subject_players(row["players"])
        sp = int(row["sp"]); qty = int(row["qty"])
        if not players or sp <= 0 or qty <= 0:
            continue
        take_max = min(qty, req.max_multiples_per_card)
        best_acct = None; best_gain = 0; best_t = 0
        for acct in FAMILY_ACCOUNTS:
            for t in range(1, take_max+1):
                sim = clone_accounts_sp(accounts)
                for _ in range(t):
                    apply_card_sp_to_account(sim[acct], players, sp, add=True, rule=req.multi_subject_rule)
                fam, _, _ = compute_family_qp(leader, sim)
                gain = fam - fam0
                if gain > best_gain:
                    best_gain = gain; best_acct = acct; best_t = t
        if best_gain > 0 and best_acct:
            suggestions.append(dict(card=row["card"], no=row["no"], players=players,
                                    assign_to=best_acct, take_n=best_t, sp=sp, family_delta_qp=best_gain))
    suggestions.sort(key=lambda x: (-x["family_delta_qp"], x["card"]))
    if req.max_each > 0:
        suggestions = suggestions[:req.max_each]
    return _safe_json({
        "family_qp_now": fam0,
        "best_buys": suggestions,
        "count": len(suggestions)
    })

@app.post("/leaderboard_delta_by_urls")
def leaderboard_delta_by_urls(req: LeaderboardDeltaReq):
    today = normalize_leaderboard(fetch_xlsx(_default(req.leaderboard_today_url, "leaderboard")))
    yday_url = body.get("leaderboard_yesterday_url") or DEFAULT_LEADERBOARD_YDAY_URL
    if not yday_url:
        raise HTTPException(status_code=400, detail="Missing yesterday URL")
    yday = normalize_leaderboard(fetch_xlsx(yday_url))

    changes = []
    players = sorted(set(list(today.keys()) + list(yday.keys())))
    rivals_set = set([_canon_user(r).lower() for r in (req.rivals or DEFAULT_RIVALS)])

    for p in players:
        t = today.get(p, [])
        y = yday.get(p, [])
        tm = {_canon_user(e["user"]): e["sp"] for e in t}
        ym = {_canon_user(e["user"]): e["sp"] for e in y}
        top3_today = [_canon_user(e["user"]) for e in t[:3]]
        top3_yday  = [_canon_user(e["user"]) for e in y[:3]]
        new_rival_top3 = [u for u in top3_today if u.lower() in rivals_set and u not in top3_yday]
        for u, tsp in tm.items():
            ysp = ym.get(u, 0)
            dsp = tsp - ysp
            if abs(dsp) >= req.min_sp_delta:
                changes.append(dict(player=p, user=u, sp_today=tsp, sp_yday=ysp, delta_sp=dsp))
        if new_rival_top3:
            changes.append(dict(player=p, new_rival_top3=new_rival_top3))

    rival_heat = Counter()
    for c in changes:
        if "user" in c and "delta_sp" in c and c["delta_sp"] > 0:
            rival_heat[c["user"]] += 1
    heat = [{"user": u, "mentions": n} for u, n in rival_heat.most_common(20)]

    return _safe_json({"changes": changes[:1000], "rival_heatmap": heat})

def is_transfer_safe(leader, accounts_sp, giver, card_players, card_sp, buffer_val) -> bool:
    fam0, per0, _ = compute_family_qp(leader, accounts_sp)
    before_qp = per0[giver]
    sim = clone_accounts_sp(accounts_sp)
    apply_card_sp_to_account(sim[giver], card_players, card_sp, add=False)
    fam1, per1, det1 = compute_family_qp(leader, sim)
    if per1[giver] < before_qp:
        return False
    frag = fragile_firsts(det1[giver], buffer_val)
    if frag:
        return False
    return True

@app.post("/family_transfer_suggestions_by_urls")
def family_transfer_suggestions_by_urls(req: FamilyTransferReq):
    leader = normalize_leaderboard(fetch_xlsx(_default(req.leaderboard_url, "leaderboard")))
    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url)
    fam0, _, _ = compute_family_qp(leader, accounts)
    col_e31 = parse_collection(fetch_xlsx(_default(req.collection_e31_url, "collection_e31")))
    col_dc  = parse_collection(fetch_xlsx(_default(req.collection_dc_url, "collection_dc")))
    col_fe  = parse_collection(fetch_xlsx(_default(req.collection_fe_url, "collection_fe")))
    collections = {
        "Easystreet31": col_e31.head(req.scan_top_candidates),
        "DusterCrusher": col_dc.head(req.scan_top_candidates),
        "FinkleIsEinhorn": col_fe.head(req.scan_top_candidates)
    }
    moves = []
    for giver in FAMILY_ACCOUNTS:
        df = collections[giver]
        for _, row in df.iterrows():
            players = split_multi_subject_players(row["players"])
            sp = int(row["sp"]); qty = int(row["qty"])
            if not players or sp <= 0 or qty <= 0:
                continue
            if not is_transfer_safe(leader, accounts, giver, players, sp, req.defend_buffer_all):
                continue
            for recv in FAMILY_ACCOUNTS:
                if recv == giver: continue
                sim = clone_accounts_sp(accounts)
                apply_card_sp_to_account(sim[giver], players, sp, add=False)
                apply_card_sp_to_account(sim[recv], players, sp, add=True)
                fam, _, _ = compute_family_qp(leader, sim)
                gain = fam - fam0
                if gain > 0:
                    moves.append(dict(from_acct=giver, to_acct=recv, card=row["card"], no=row["no"],
                                      players=players, move_n=1, sp=sp, family_delta_qp=gain))
    moves.sort(key=lambda x: (-x["family_delta_qp"], x["card"]))
    out_moves = moves[:req.limit] if req.limit > 0 else moves
    fam_after = fam0 + sum(m["family_delta_qp"] for m in out_moves)
    return _safe_json({
        "family_qp_now": fam0,
        "family_qp_after_naive_sum": fam_after,
        "internal_transfers": out_moves,
        "note": "Naive sum; for exact combined plan, use /family_transfer_optimize_by_urls"
    })

@app.post("/family_transfer_optimize_by_urls")
def family_transfer_optimize_by_urls(req: FamilyTransferReq):
    if not HAS_ORTOOLS:
        return family_transfer_suggestions_by_urls(req)
    leader = normalize_leaderboard(fetch_xlsx(_default(req.leaderboard_url, "leaderboard")))
    accounts = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url)
    fam0, _, _ = compute_family_qp(leader, accounts)
    col_e31 = parse_collection(fetch_xlsx(_default(req.collection_e31_url, "collection_e31")))
    col_dc  = parse_collection(fetch_xlsx(_default(req.collection_dc_url, "collection_dc")))
    col_fe  = parse_collection(fetch_xlsx(_default(req.collection_fe_url, "collection_fe")))
    collections = {
        "Easystreet31": col_e31.head(req.scan_top_candidates),
        "DusterCrusher": col_dc.head(req.scan_top_candidates),
        "FinkleIsEinhorn": col_fe.head(req.scan_top_candidates)
    }
    # Build candidates
    unit_keys = []
    candidates = []
    for giver in FAMILY_ACCOUNTS:
        df = collections[giver]
        for _, row in df.iterrows():
            players = split_multi_subject_players(row["players"])
            sp = int(row["sp"]); qty = int(row["qty"])
            if not players or sp <= 0 or qty <= 0:
                continue
            if not is_transfer_safe(leader, accounts, giver, players, sp, req.defend_buffer_all):
                continue
            for u in range(min(qty, req.max_multiples_per_card)):
                unit_keys.append((giver, row["card"], row["no"], u, players, sp))
                for recv in FAMILY_ACCOUNTS:
                    if recv == giver: continue
                    sim = clone_accounts_sp(accounts)
                    apply_card_sp_to_account(sim[giver], players, sp, add=False)
                    apply_card_sp_to_account(sim[recv], players, sp, add=True)
                    fam, _, _ = compute_family_qp(leader, sim)
                    gain = fam - fam0
                    if gain > 0:
                        candidates.append({
                            "unit_idx": len(unit_keys)-1, "giver": giver, "receiver": recv,
                            "card": row["card"], "no": row["no"],
                            "players": players, "sp": sp, "gain": gain
                        })
    model = cp_model.CpModel()
    y = {i: model.NewBoolVar(f"y_{i}") for i in range(len(candidates))}
    by_unit: Dict[int, List[int]] = defaultdict(list)
    for i, cand in enumerate(candidates):
        by_unit[cand["unit_idx"]].append(i)
    for unit_idx, inds in by_unit.items():
        model.Add(sum(y[i] for i in inds) <= 1)
    model.Maximize(sum(int(candidates[i]["gain"]*1000) * y[i] for i in range(len(candidates))))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 8
    res = solver.Solve(model)
    chosen = [candidates[i] for i in range(len(candidates)) if solver.Value(y[i]) == 1] if res in (cp_model.OPTIMAL, cp_model.FEASIBLE) else []
    sim = clone_accounts_sp(accounts)
    for c in chosen:
        apply_card_sp_to_account(sim[c["giver"]], c["players"], c["sp"], add=False)
        apply_card_sp_to_account(sim[c["receiver"]], c["players"], c["sp"], add=True)
    fam1, _, _ = compute_family_qp(leader, sim)
    return _safe_json({
        "family_qp_now": fam0,
        "family_qp_after": fam1,
        "delta_qp": fam1 - fam0,
        "moves": chosen,
        "note": "Exact ILP plan over single-copy moves"
    })
