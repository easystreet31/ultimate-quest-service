# main.py
# Ultimate Quest Service — v3.3.0 "partner-safe-give"
#
# New endpoint:
#   POST /suggest_give_from_collection_by_urls_easystreet31
#     - Finds cards in *your* collection you can trade TO a partner
#     - Partner gains QP; you do NOT lose QP; optional 1st-place buffer protection
#
# Existing helper endpoints remain:
#   GET  /health
#   GET  /info
#
# Notes:
#  - Robust XLSX readers
#  - Multi-subject logic: "full_each_unique"
#  - Conservative tie-handling: must exceed (>) competitor SP to beat their rank
#  - NaN/Inf scrubbed before JSON
#
# Requirements: fastapi, uvicorn, pandas, openpyxl, python-multipart, requests

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import requests, io, math

APP_VERSION = "3.3.0-partner-safe-give"

app = FastAPI(title="Ultimate Quest Service", version=APP_VERSION)


# ---------------------------
# Utilities
# ---------------------------

def _fetch_xlsx(url: str) -> Dict[str, pd.DataFrame]:
    """Download an xlsx (Google Sheets export) and return {sheetname: DataFrame}."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = io.BytesIO(resp.content)
    xl = pd.ExcelFile(data)
    sheets = {}
    for name in xl.sheet_names:
        try:
            df = xl.parse(name)
            if isinstance(df, pd.DataFrame) and df.shape[0] > 0:
                sheets[name] = df
        except Exception:
            continue
    if not sheets:
        raise ValueError("No readable sheets found in XLSX.")
    return sheets


def _first_sheet_with(df_map: Dict[str, pd.DataFrame], needed_cols: List[str]) -> pd.DataFrame:
    """Pick the first sheet that contains all needed columns (case-insensitive fuzzy)."""
    def cols(d: pd.DataFrame) -> List[str]:
        return [str(c).strip() for c in d.columns]

    for _, df in df_map.items():
        low = {c.lower().strip(): c for c in cols(df)}
        ok = True
        for n in needed_cols:
            if not any(n in k for k in low.keys()):
                ok = False
                break
        if ok:
            return df
    # else return the first sheet
    return list(df_map.values())[0]


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _norm_user(u: Union[str, float, int]) -> str:
    s = str(u or "").strip()
    # strip trailing " (1234)"
    if " (" in s and s.endswith(")"):
        s = s[: s.rfind(" (")].strip()
    return s


def _qp_for_rank(rank: Optional[int]) -> int:
    if rank == 1: return 5
    if rank == 2: return 3
    if rank == 3: return 1
    return 0


def _split_players(players_str: str) -> List[str]:
    if not isinstance(players_str, str):
        return []
    # split on common separators
    parts = [p.strip() for p in
             re_split(players_str, [",", "&", "/", "+", ";", "|"])]
    # drop empties and de-dup while preserving order
    seen = set(); out = []
    for p in parts:
        if p and p.lower() not in seen:
            out.append(p)
            seen.add(p.lower())
    return out


def re_split(text: str, seps: List[str]) -> List[str]:
    import re
    pattern = "|".join(map(re_escape, seps))
    return re.split(pattern, text)


def re_escape(s: str) -> str:
    import re
    return re.escape(s)


def _clean_json(obj: Any) -> Any:
    """Make everything JSON-safe (no NaN/Inf)."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if obj is None:
        return None
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# ---------------------------
# Parsers
# ---------------------------

def parse_leaderboard(url: str) -> pd.DataFrame:
    """
    Expect a long format with columns similar to:
      player, rank, username/user, sp[, qp]
    We'll normalize to: player, user, rank, sp, qp
    Only keep rank <= 5.
    """
    sheets = _fetch_xlsx(url)
    # try to pick a sheet that has player, rank, user, sp
    df = _first_sheet_with(sheets, ["player", "rank", "sp"])
    df = _norm_cols(df)

    # map possible aliases
    col_map = {
        "player": [ "player", "subject", "name" ],
        "rank":   [ "rank", "rnk" ],
        "user":   [ "username", "user", "collector", "owner" ],
        "sp":     [ "sp", "subject_points", "points" ],
        "qp":     [ "qp", "quest_points" ],
    }
    def pick(df, keys):
        for k in keys:
            for c in df.columns:
                if c == k or c.startswith(k):
                    return c
        return None

    c_player = pick(df, col_map["player"])
    c_rank   = pick(df, col_map["rank"])
    c_user   = pick(df, col_map["user"])
    c_sp     = pick(df, col_map["sp"])
    c_qp     = pick(df, col_map["qp"])

    if not all([c_player, c_rank, c_user, c_sp]):
        raise ValueError("Leaderboard sheet missing required columns (player, rank, user, sp).")

    out = df[[c_player, c_rank, c_user, c_sp] + ([c_qp] if c_qp else [])].copy()
    out.rename(columns={c_player:"player", c_rank:"rank", c_user:"user", c_sp:"sp"}, inplace=True)
    if c_qp:
        out.rename(columns={c_qp:"qp"}, inplace=True)
    out["user"]  = out["user"].map(_norm_user)
    out["rank"]  = pd.to_numeric(out["rank"], errors="coerce").astype("Int64")
    out["sp"]    = pd.to_numeric(out["sp"], errors="coerce").fillna(0).astype(int)
    if "qp" in out.columns:
        out["qp"] = pd.to_numeric(out["qp"], errors="coerce").fillna(0).astype(int)
    else:
        out["qp"] = out["rank"].map(lambda r: _qp_for_rank(int(r)) if pd.notna(r) else 0)

    out = out[out["rank"].fillna(99) <= 5].copy()
    out["player"] = out["player"].astype(str).str.strip()
    return out.reset_index(drop=True)


def parse_holdings(url: str, me: str = "Easystreet31") -> pd.DataFrame:
    """
    Expect a table with (at least):
      player, sp (mine), rank (mine), qp (mine)
    We'll normalize to: player, you_sp, rank, you_qp
    """
    sheets = _fetch_xlsx(url)
    df = _first_sheet_with(sheets, ["player", "sp"])
    df = _norm_cols(df)

    # attempt to detect my columns
    c_player = [c for c in df.columns if c.startswith("player") or c.startswith("subject")]
    c_sp     = [c for c in df.columns if c in ["you_sp", "sp", "my_sp", "easystreet31_sp"]]
    c_rank   = [c for c in df.columns if c in ["rank", "my_rank", "you_rank"]]
    c_qp     = [c for c in df.columns if c in ["qp", "my_qp", "you_qp"]]

    if not c_player or not c_sp:
        raise ValueError("Holdings sheet missing required columns (player, sp).")

    out = pd.DataFrame()
    out["player"] = df[c_player[0]].astype(str).str.strip()
    out["you_sp"] = pd.to_numeric(df[c_sp[0]], errors="coerce").fillna(0).astype(int)
    if c_rank:
        out["rank"] = pd.to_numeric(df[c_rank[0]], errors="coerce").astype("Int64")
    else:
        out["rank"] = pd.NA
    if c_qp:
        out["you_qp"] = pd.to_numeric(df[c_qp[0]], errors="coerce").fillna(0).astype(int)
    else:
        out["you_qp"] = out["rank"].map(lambda r: _qp_for_rank(int(r)) if pd.notna(r) else 0)

    return out.reset_index(drop=True)


def parse_my_collection(url: str) -> pd.DataFrame:
    """
    Expect a card-level table with columns (case-insensitive, synonyms accepted):
      - card name / title
      - card number (optional)
      - players or subject(s)
      - sp (subject points)
      - qty / quantity / count
    """
    sheets = _fetch_xlsx(url)
    df = _first_sheet_with(sheets, ["players", "sp"])
    df = _norm_cols(df)

    # flexible column discovery
    def find1(names):
        for n in names:
            for c in df.columns:
                if c == n or c.startswith(n):
                    return c
        return None

    c_players = find1(["players", "subjects", "player", "subject"])
    c_sp      = find1(["sp", "subject_points", "points"])
    c_qty     = find1(["qty", "quantity", "count", "copies", "q"])
    c_card    = find1(["card_name", "card", "title"])
    c_no      = find1(["card_number", "card_no", "no", "number"])

    if not c_players or not c_sp:
        raise ValueError("My Collection sheet needs at least 'Players' and 'SP' columns.")

    out = pd.DataFrame()
    out["card_name"]   = df[c_card].astype(str).str.strip() if c_card else ""
    out["card_number"] = df[c_no].astype(str).str.strip() if c_no else ""
    out["players_raw"] = df[c_players].astype(str).str.strip()
    out["sp"]          = pd.to_numeric(df[c_sp], errors="coerce").fillna(0).astype(int)
    out["qty"]         = pd.to_numeric(df[c_qty], errors="coerce").fillna(1).astype(int) if c_qty else 1
    return out.reset_index(drop=True)


# ---------------------------
# Ranking & simulation
# ---------------------------

def leaderboard_map(lb: pd.DataFrame) -> Dict[str, List[Tuple[str,int,int]]]:
    """
    Returns: {player: [(user, sp, rank), ...] sorted by rank asc}
    """
    d: Dict[str, List[Tuple[str,int,int]]] = {}
    for player, grp in lb.groupby("player"):
        recs = []
        for _, r in grp.sort_values(["rank", "sp"], ascending=[True, False]).iterrows():
            recs.append((str(r["user"]), int(r["sp"]), int(r["rank"])))
        d[player] = recs
    return d


def current_sp_of(user: str, player: str, lb_map: Dict[str, List[Tuple[str,int,int]]]) -> int:
    arr = lb_map.get(player, [])
    for u, sp, _ in arr:
        if _norm_user(u).lower() == _norm_user(user).lower():
            return sp
    return 0


def simulate_insert_rank(
    arr: List[Tuple[str,int,int]],
    user: str,
    new_sp: int
) -> Tuple[int, int, List[Tuple[str,int]]]:
    """
    Given current top rows for a player: [(user, sp, rank), ...], insert (user,new_sp)
    and compute new rank for 'user'. Conservative tie logic: competitor keeps place on ties.
    Returns: (new_rank, margin_if_first_else_gap_to_next, new_order_as[(user,sp)])
    """
    base = [(u, sp) for (u, sp, _) in arr]
    # remove if exists
    base = [(u, sp) for (u, sp) in base if _norm_user(u).lower() != _norm_user(user).lower()]
    # tie-breaker: original index
    order_index = {u: i for i, (u, _) in enumerate(base)}
    base.append((user, new_sp))

    # sort: by sp desc; then by original order (competitors first), then put 'user' last among equals
    def sort_key(item):
        u, sp = item
        tie = order_index.get(u, 1_000_000)
        return (-sp, tie)

    sorted_list = sorted(base, key=sort_key)
    # ranks assigned by position
    new_rank = 1 + [u for (u, _) in sorted_list].index(user)

    # compute margin if first else gap to the next better place
    def margin_or_gap(lst):
        idx = [u for (u, _) in lst].index(user)
        if idx == 0:
            # margin vs second (or 0 if none)
            return lst[0][1] - (lst[1][1] if len(lst) > 1 else 0)
        else:
            return lst[idx-1][1] - lst[idx][1]  # positive gap needed to move up
    mg = margin_or_gap(sorted_list)

    return new_rank, mg, sorted_list


def qp_for(user: str, player: str, lb_map: Dict[str, List[Tuple[str,int,int]]], sp: int) -> Tuple[int,int,int,int]:
    """
    Given a target sp, compute (rank, qp, margin_if_first_else_gap, second_sp).
    """
    arr = lb_map.get(player, [])
    new_rank, mg, new_sorted = simulate_insert_rank(arr, user, sp)
    qp = _qp_for_rank(new_rank)
    # find second sp for margin display
    second_sp = new_sorted[1][1] if len(new_sorted) > 1 else 0
    return new_rank, qp, mg, second_sp


# ---------------------------
# Partner Safe Give
# ---------------------------

class SuggestGiveReq(BaseModel):
    leaderboard_url: str
    holdings_url: str
    my_collection_url: str
    partner: str

    # knobs
    multi_subject_rule: str = Field("full_each_unique", description="multi-subject scoring")
    protect_qp: bool = True
    protect_buffer: int = 20
    max_each: int = 50
    max_multiples_per_card: int = 3
    players_whitelist: Optional[List[str]] = None
    players_blacklist: Optional[List[str]] = None
    target_rivals: Optional[List[str]] = None
    rival_score_weight: int = 250  # bump scoring if we dethrone a target rival

class Suggestion(BaseModel):
    card: str
    card_number: str
    players: List[str]
    sp: int
    take_n: int
    my_qp_change: int
    partner_qp_change: int
    my_impacts: List[Dict[str, Any]]
    partner_impacts: List[Dict[str, Any]]
    rival_impacts: List[Dict[str, Any]]
    score: int


@app.post("/suggest_give_from_collection_by_urls_easystreet31")
def suggest_give_from_collection(req: SuggestGiveReq):
    try:
        me = "Easystreet31"
        partner = _norm_user(req.partner)

        # Load data
        lb = parse_leaderboard(req.leaderboard_url)
        my = parse_holdings(req.holdings_url, me=me)
        coll = parse_my_collection(req.my_collection_url)

        # map by player
        lb_map = leaderboard_map(lb)
        my_map = {r["player"]: {"you_sp": int(r["you_sp"]), "rank": int(r["rank"]) if pd.notna(r["rank"]) else None,
                                "you_qp": int(r["you_qp"])} for _, r in my.iterrows()}

        # helper for filtering players
        allow = None
        if req.players_whitelist:
            wl = set([p.lower().strip() for p in req.players_whitelist])
            allow = lambda p: p.lower().strip() in wl
        elif req.players_blacklist:
            bl = set([p.lower().strip() for p in req.players_blacklist])
            allow = lambda p: p.lower().strip() not in bl
        else:
            allow = lambda p: True

        suggestions: List[Suggestion] = []
        diag_errors: List[Dict[str, Any]] = []

        # iterate cards
        for _, row in coll.iterrows():
            card_name = str(row.get("card_name", "") or "").strip()
            card_no   = str(row.get("card_number", "") or "").strip()
            sp        = int(row.get("sp", 0) or 0)
            qty       = int(row.get("qty", 1) or 1)

            if sp <= 0 or qty <= 0:
                continue

            players = _split_players(row.get("players_raw", ""))
            if req.multi_subject_rule == "full_each_unique":
                players = list(dict.fromkeys(players))  # unique, preserve order
            else:
                players = players[:1]  # safest fallback

            # Apply whitelist/blacklist
            players = [p for p in players if allow(p)]
            if not players:
                continue

            # Compute max copies we can safely give without hurting QP/buffer
            # For each player, compute max copies allowed
            max_copies_allowed = qty
            for p in players:
                mine = my_map.get(p, {"you_sp":0, "rank":None, "you_qp":0})
                cur_sp = int(mine["you_sp"] or 0)
                cur_rank = int(mine["rank"]) if mine["rank"] is not None else None
                cur_qp = int(mine["you_qp"] or 0)

                if req.protect_qp:
                    # test copies from 0..qty and find largest safe n
                    safe_n = 0
                    for n in range(0, min(qty, req.max_multiples_per_card)+1):
                        new_sp = max(0, cur_sp - n*sp)
                        rnk, qp, mg, sec_sp = qp_for(me, p, lb_map, new_sp)
                        if qp < cur_qp:
                            break
                        if cur_qp == 5 and req.protect_buffer:
                            # when first, ensure margin >= protect_buffer
                            # margin computed as new_sp - second_sp
                            if (new_sp - sec_sp) < req.protect_buffer:
                                break
                        safe_n = n
                    max_copies_allowed = min(max_copies_allowed, safe_n)
                else:
                    # only enforce non-negative SP
                    safe_n = min(qty, req.max_multiples_per_card)
                    max_copies_allowed = min(max_copies_allowed, safe_n)

            max_copies_allowed = min(max_copies_allowed, req.max_multiples_per_card)
            if max_copies_allowed <= 0:
                continue

            # Evaluate 1..max_copies_allowed
            for take_n in range(1, max_copies_allowed+1):
                my_total_qp_delta = 0
                partner_total_qp_delta = 0
                my_imp, partner_imp = [], []
                rival_imp = []
                score = 0

                # Evaluate per player effect
                for p in players:
                    mine = my_map.get(p, {"you_sp":0, "rank":None, "you_qp":0})
                    my_sp_before = int(mine["you_sp"] or 0)
                    my_rank_b = int(mine["rank"]) if mine["rank"] is not None else None
                    my_qp_b = int(mine["you_qp"] or 0)

                    # Me after
                    my_sp_after = max(0, my_sp_before - take_n*sp)
                    my_rank_a, my_qp_a, my_mg_a, my_sec_after = qp_for(me, p, lb_map, my_sp_after)
                    my_total_qp_delta += (my_qp_a - my_qp_b)

                    my_imp.append({
                        "player": p,
                        "my_sp": my_sp_before,
                        "my_sp_after": my_sp_after,
                        "rank_before": my_rank_b,
                        "rank_after": my_rank_a,
                        "qp_before": my_qp_b,
                        "qp_after": my_qp_a,
                        "margin_if_first_after": (my_sp_after - my_sec_after) if my_rank_a == 1 else None
                    })

                    # Partner before/after
                    part_sp_b = current_sp_of(partner, p, lb_map)  # 0 if not in top-5
                    part_rank_b, part_qp_b, _, _ = qp_for(partner, p, lb_map, part_sp_b)
                    part_sp_a = part_sp_b + take_n*sp
                    part_rank_a, part_qp_a, _, _ = qp_for(partner, p, lb_map, part_sp_a)
                    partner_total_qp_delta += (part_qp_a - part_qp_b)

                    partner_imp.append({
                        "player": p,
                        "partner_sp": part_sp_b,
                        "partner_sp_after": part_sp_a,
                        "rank_before": part_rank_b,
                        "rank_after": part_rank_a,
                        "qp_before": part_qp_b,
                        "qp_after": part_qp_a
                    })

                    # rival impact: if the dethroned user is in target_rivals, add weight
                    if req.target_rivals:
                        rivals = set([_norm_user(x).lower() for x in req.target_rivals])
                        # if partner enters top-3 or improves rank over someone in rivals
                        arr = lb_map.get(p, [])
                        # who owned the place partner is taking?
                        if part_qp_a > part_qp_b:
                            # we can infer the loser: rank that decreased among existing records
                            # Approx: if partner enters 3rd, previous 3rd drops to 4th -> identify that user
                            # We'll simply check: among current top3, any rival whose SP < partner_sp_after and whose rank >= part_rank_a
                            affected = []
                            for u, sp_old, r_old in arr:
                                u0 = _norm_user(u).lower()
                                if u0 in rivals and sp_old < part_sp_a and r_old >= part_rank_a and r_old <= 3:
                                    affected.append(u)
                            if affected:
                                for rv in affected:
                                    rival_imp.append({"player": p, "rival": _norm_user(rv), "note": "partner overtakes rival in top-3"})
                                    score += req.rival_score_weight

                # Enforce "do not hurt me"
                if req.protect_qp and my_total_qp_delta < 0:
                    continue

                # Only keep if partner gains QP
                if partner_total_qp_delta <= 0:
                    continue

                score += partner_total_qp_delta * 1000  # primary sort key
                score += -abs(my_total_qp_delta) * 100  # small bias to zero-change

                suggestions.append(Suggestion(
                    card=card_name or "(unnamed)",
                    card_number=card_no,
                    players=players,
                    sp=sp,
                    take_n=take_n,
                    my_qp_change=int(my_total_qp_delta),
                    partner_qp_change=int(partner_total_qp_delta),
                    my_impacts=my_imp,
                    partner_impacts=partner_imp,
                    rival_impacts=rival_imp,
                    score=int(score)
                ))

        # order & truncate
        suggestions.sort(key=lambda s: (-s.score, -s.partner_qp_change, s.card, -s.take_n))
        omitted = max(0, len(suggestions) - max(0, req.max_each))
        if req.max_each > 0:
            suggestions = suggestions[:req.max_each]

        result = {
            "params": {
                "multi_subject_rule": req.multi_subject_rule,
                "protect_qp": req.protect_qp,
                "protect_buffer": req.protect_buffer,
                "max_each": req.max_each,
                "max_multiples_per_card": req.max_multiples_per_card,
                "players_whitelist": req.players_whitelist or [],
                "players_blacklist": req.players_blacklist or [],
                "partner": partner,
                "target_rivals": req.target_rivals or [],
                "rival_score_weight": req.rival_score_weight
            },
            "my_collection_counts": {
                "card_types": int(coll.shape[0]),
                "total_units": int(coll["qty"].fillna(1).astype(int).sum())
            },
            "safe_gives": [s.dict() for s in suggestions],
            "omitted": int(omitted)
        }
        return _clean_json(result)

    except Exception as e:
        return {"detail": f"Partner safe-give failed: {e}"}


# ---------------------------
# Health / Info
# ---------------------------

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
            "/health",
            "/info",
            "/suggest_give_from_collection_by_urls_easystreet31"
        ]
    }
