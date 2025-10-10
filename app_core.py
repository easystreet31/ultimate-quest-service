# app_core.py (v5.0)
"""
Ultimate Quest Service Core Business Logic
- Leaderboard normalization
- Trade evaluation with 5-3-1 QP scoring
- Rank/buffer calculations
- Delta reporting
"""

import io
import typing as t
from collections import defaultdict
from datetime import datetime
import time
import requests

import pandas as pd
from fastapi import HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field, validator

import config
import logging_utils as log_util
import cache_utils

logger = log_util.get_logger("app_core")


# ============================================================================
# Constants & Mappings
# ============================================================================

FAMILY_ACCOUNTS = config.FAMILY_ACCOUNTS
QP_MAP = config.QP_MAP
SYNDICATE = config.SYNDICATE


def _strip_user_suffix(s: t.Any) -> str:
    """Remove usernames in parentheses, e.g. 'Player (Owner)' -> 'Player'."""
    s = str(s or "")
    s = s.split("\n", 1)[0]
    s = s.split("(", 1)[0]
    return s.strip()


def _norm_name(s: t.Any) -> str:
    """Normalize a display name: strip whitespace."""
    return " ".join(str(s or "").split()).strip()


def _norm_key(s: t.Any) -> str:
    """Normalize a key for lookup: lowercase, remove extra spaces."""
    return _norm_name(s).lower()


def _canon_key(s: t.Any) -> str:
    """Canonical key: remove all non-alphanumeric, lowercase."""
    raw = "".join(ch for ch in str(s or "") if ch.isalnum())
    return raw.lower()


_FAMILY_CANON_TO_DISPLAY: t.Dict[str, str] = {
    _canon_key(a): a for a in FAMILY_ACCOUNTS
}
_FAMILY_KEYS: t.Set[str] = set(_FAMILY_CANON_TO_DISPLAY.keys())


# ============================================================================
# HTTP & Data Fetching (with retry + cache)
# ============================================================================

def _http_get_bytes_with_retry(url: str) -> bytes:
    """Fetch URL with exponential backoff retry logic."""
    headers = {
        "User-Agent": config.REQUESTS_USER_AGENT,
        "Accept": "*/*"
    }
    
    last_error = None
    for attempt in range(1, config.REQUEST_RETRY_MAX_ATTEMPTS + 1):
        try:
            logger.debug(f"Fetching URL (attempt {attempt})", extra={"url": url})
            r = requests.get(
                url,
                headers=headers,
                timeout=config.REQUEST_TIMEOUT_SECONDS
            )
            r.raise_for_status()
            logger.info(f"Successfully fetched URL", extra={"url": url, "size_bytes": len(r.content)})
            return r.content
        
        except requests.Timeout:
            last_error = f"Timeout after {config.REQUEST_TIMEOUT_SECONDS}s"
            if attempt < config.REQUEST_RETRY_MAX_ATTEMPTS:
                wait = config.REQUEST_RETRY_INITIAL_WAIT * (config.REQUEST_RETRY_BACKOFF_FACTOR ** (attempt - 1))
                logger.warning(
                    f"Timeout, retrying in {wait}s",
                    extra={"url": url, "attempt": attempt, "wait_seconds": wait}
                )
                time.sleep(wait)
        
        except requests.HTTPError as e:
            last_error = f"HTTP {e.response.status_code}"
            if e.response.status_code >= 500 and attempt < config.REQUEST_RETRY_MAX_ATTEMPTS:
                wait = config.REQUEST_RETRY_INITIAL_WAIT * (config.REQUEST_RETRY_BACKOFF_FACTOR ** (attempt - 1))
                logger.warning(
                    f"Server error, retrying in {wait}s",
                    extra={"url": url, "status": e.response.status_code, "attempt": attempt}
                )
                time.sleep(wait)
            else:
                logger.error(f"HTTP error", extra={"url": url, "status": e.response.status_code})
                raise
        
        except Exception as e:
            logger.error(
                f"Unexpected error fetching URL",
                exc_info=True,
                extra={"url": url, "error": str(e)}
            )
            raise
    
    # All retries exhausted
    error_msg = f"Failed to fetch after {config.REQUEST_RETRY_MAX_ATTEMPTS} attempts: {last_error}"
    logger.error(error_msg, extra={"url": url})
    raise HTTPException(
        status_code=502,
        detail=f"Failed to fetch external sheet: {last_error}"
    )


@cache_utils.cached_url_request
def fetch_xlsx(url: str) -> t.Dict[str, pd.DataFrame]:
    """Fetch and parse XLSX, with caching."""
    raw = _http_get_bytes_with_retry(url)
    
    try:
        with pd.ExcelFile(io.BytesIO(raw)) as xf:
            sheets = {}
            for name in xf.sheet_names:
                df = xf.parse(name)
                df.columns = [str(c).strip() for c in df.columns]
                sheets[name] = df
            
            logger.info(
                f"Parsed XLSX successfully",
                extra={"url": url, "sheets": len(sheets), "sheet_names": list(sheets.keys())}
            )
            return sheets
    
    except Exception as e:
        logger.error(
            f"Failed to parse XLSX",
            exc_info=True,
            extra={"url": url, "error": str(e)}
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse sheet: {str(e)}"
        )


# ============================================================================
# Column Detection (with validation)
# ============================================================================

def _detect_col(df: pd.DataFrame, candidates: t.Iterable[str]) -> str:
    """Detect a column by name, case-insensitive."""
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    
    for want in candidates:
        lw = want.lower()
        if lw in lower_map:
            return lower_map[lw]
    
    # Fallback: look for keyword matches
    for c in df.columns:
        cl = str(c).lower().strip()
        if any(tok in cl for tok in ("username", "user_name", "handle")):
            if "user" in [w.lower() for w in candidates]:
                return c
        if any(tok in cl for tok in ("account_name", "acct", "acct_name")):
            if "account" in [w.lower() for w in candidates]:
                return c
    
    raise KeyError(f"Missing required column (tried {list(candidates)}) in: {list(df.columns)}")


def _detect_col_qp(df: pd.DataFrame) -> t.Optional[str]:
    """Detect QP column, optional."""
    for base in ("qp", "quest", "quest_points"):
        try:
            return _detect_col(df, [base])
        except KeyError:
            pass
    
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl == "qp":
            return c
        if "quest" in cl and any(x in cl for x in ("point", "total", "qp")):
            return c
        if "qp" in cl:
            return c
    
    return None


# ============================================================================
# Leaderboard Normalization
# ============================================================================

class _Row(t.TypedDict, total=False):
    account: str
    sp: int
    qp: int
    rank: t.Optional[int]


class _Leader(t.TypedDict):
    by_player: t.Dict[str, t.List[_Row]]
    sp_map: t.Dict[str, t.Dict[str, int]]
    display_names: t.Dict[str, str]
    account_display: t.Dict[str, str]


@log_util.timed_operation("normalize_leaderboard")
def normalize_leaderboard(sheets: t.Dict[str, pd.DataFrame]) -> "_Leader":
    """
    Parse leaderboard sheet and return normalized structure.
    Validates required columns exist.
    """
    if not sheets:
        raise ValueError("No sheets provided")
    
    name = next(iter(sheets))
    df = sheets[name].copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    # Detect required columns
    col_player = _detect_col(df, ["player", "players", "subject", "name"])
    col_account = _detect_col(df, ["account", "username", "user", "owner", "handle"])
    col_sp = _detect_col(df, ["sp", "score", "points"])
    col_qp = _detect_col_qp(df)
    
    col_rank = None
    for cand in ("rank", "position"):
        try:
            col_rank = _detect_col(df, [cand])
            break
        except KeyError:
            pass
    
    logger.debug(
        "Detected columns",
        extra={
            "player": col_player,
            "account": col_account,
            "sp": col_sp,
            "qp": col_qp,
            "rank": col_rank
        }
    )
    
    # Normalize data
    df[col_player] = df[col_player].map(_norm_name)
    df[col_account] = df[col_account].map(_strip_user_suffix).map(_norm_name)
    df[col_sp] = pd.to_numeric(df[col_sp], errors="coerce").fillna(0).astype(int)
    
    if col_qp:
        df[col_qp] = pd.to_numeric(df[col_qp], errors="coerce").fillna(0).astype(int)
    if col_rank:
        df[col_rank] = pd.to_numeric(df[col_rank], errors="coerce").fillna(0).astype(int)
    
    by_player: t.Dict[str, t.List[_Row]] = defaultdict(list)
    sp_map: t.Dict[str, t.Dict[str, int]] = defaultdict(dict)
    display_player: t.Dict[str, str] = {}
    account_display: t.Dict[str, str] = {}
    
    for _, r in df.iterrows():
        disp_p = _norm_name(r[col_player])
        p = _norm_key(disp_p)
        acc_disp = _norm_name(r[col_account])
        acc_canon = _canon_key(acc_disp)
        sp = int(r[col_sp]) if pd.notna(r[col_sp]) else 0
        qp = int(r[col_qp]) if col_qp and pd.notna(r[col_qp]) else 0
        rk = int(r[col_rank]) if col_rank and pd.notna(r[col_rank]) else None
        
        display_player.setdefault(p, disp_p)
        account_display.setdefault(acc_canon, acc_disp)
        
        by_player[p].append(_Row(account=acc_canon, sp=sp, qp=qp, rank=rk))
        prev = int(sp_map[p].get(acc_canon, 0))
        sp_map[p][acc_canon] = max(prev, sp)
    
    result: _Leader = {
        "by_player": dict(by_player),
        "sp_map": dict(sp_map),
        "display_names": display_player,
        "account_display": account_display,
    }
    
    logger.info(
        "Leaderboard normalized",
        extra={"players": len(display_player), "accounts": len(account_display)}
    )
    
    return result


# ============================================================================
# Holdings & Tags
# ============================================================================

def _frame_to_holdings(df: pd.DataFrame) -> t.Dict[str, int]:
    """Convert DataFrame to {player -> sp} dict."""
    df.columns = [str(c).strip() for c in df.columns]
    col_player = _detect_col(df, ["player", "players", "subject", "name"])
    
    col_sp = None
    for c in ("sp", "score", "points", "count", "qty", "quantity"):
        try:
            col_sp = _detect_col(df, [c])
            break
        except KeyError:
            pass
    
    if not col_sp:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            logger.warning("No numeric columns found for holdings")
            return {}
        col_sp = numeric_cols[0]
        logger.debug(f"Auto-detected SP column: {col_sp}")
    
    df = df[[col_player, col_sp]].copy()
    df[col_player] = df[col_player].map(_norm_name)
    df[col_sp] = pd.to_numeric(df[col_sp], errors="coerce").fillna(0).astype(int)
    
    grp = df.groupby(col_player, dropna=False)[col_sp].sum().reset_index()
    return {_norm_key(r[col_player]): int(r[col_sp]) for _, r in grp.iterrows()}


@log_util.timed_operation("load_holdings")
def holdings_from_urls(
    e31: t.Optional[str],
    dc: t.Optional[str],
    fe: t.Optional[str],
    prefer_env_defaults: bool,
    ud: t.Optional[str]
) -> t.Dict[str, t.Dict[str, int]]:
    """Load holdings for all family accounts."""
    urls = {
        "Easystreet31": _pick_url(e31, "holdings_e31", prefer_env_defaults),
        "DusterCrusher": _pick_url(dc, "holdings_dc", prefer_env_defaults),
        "FinkleIsEinhorn": _pick_url(fe, "holdings_fe", prefer_env_defaults),
        "UpperDuck": _pick_url(ud, "holdings_ud", prefer_env_defaults),
    }
    
    out: t.Dict[str, t.Dict[str, int]] = {a: {} for a in FAMILY_ACCOUNTS}
    
    for acct, url in urls.items():
        try:
            raw = _http_get_bytes_with_retry(url)
            with pd.ExcelFile(io.BytesIO(raw)) as xf:
                df = xf.parse(xf.sheet_names[0])
                out[acct] = _frame_to_holdings(df)
                logger.info(f"Loaded holdings for {acct}", extra={"count": len(out[acct])})
        except Exception as e:
            logger.error(
                f"Failed to load holdings for {acct}",
                exc_info=True,
                extra={"url": url, "error": str(e)}
            )
            raise
    
    return out


@log_util.timed_operation("load_player_tags")
def _load_player_tags(prefer_env_defaults: bool, player_tags_url: t.Optional[str]) -> t.Dict[str, t.Set[str]]:
    """Load player tags from sheet, with consistent normalization."""
    url = _pick_url(player_tags_url, "player_tags", prefer_env_defaults)
    raw = _http_get_bytes_with_retry(url)
    
    tags: t.Dict[str, t.Set[str]] = {cat: set() for cat in config.PLAYER_TAG_CATEGORIES}
    
    try:
        with pd.ExcelFile(io.BytesIO(raw)) as xf:
            for tab, key in [
                ("Legends", "LEGENDS"),
                ("ANA", "ANA"),
                ("DAL", "DAL"),
                ("LAK", "LAK"),
                ("PIT", "PIT")
            ]:
                if tab not in xf.sheet_names:
                    logger.debug(f"Tab not found in player_tags sheet: {tab}")
                    continue
                
                df = xf.parse(tab)
                if df.empty:
                    logger.debug(f"Tab {tab} is empty")
                    continue
                
                col = df.columns[0]
                vals = [_norm_key(v) for v in list(df[col].astype(str)) if str(v).strip()]
                tags[key] |= set(vals)
                logger.debug(f"Loaded {len(vals)} players for tag {key}")
        
        total_tagged = sum(len(v) for v in tags.values())
        logger.info(f"Player tags loaded", extra={"total_players": total_tagged, "tags": list(tags.keys())})
    
    except Exception as e:
        logger.error(
            f"Failed to load player tags",
            exc_info=True,
            extra={"url": url, "error": str(e)}
        )
        raise
    
    return tags


# ============================================================================
# URL Handling
# ============================================================================

def _pick_url(explicit: t.Optional[str], kind: str, prefer_env_defaults: bool) -> str:
    """Pick a URL from explicit param, env var, or raise."""
    if explicit and explicit.strip():
        return explicit.strip()
    
    if not prefer_env_defaults:
        raise ValueError(f"No explicit URL for '{kind}' and prefer_env_defaults is False")
    
    try:
        return config.get_env_url(kind)
    except ValueError as e:
        logger.error(f"Missing URL configuration", extra={"kind": kind})
        raise


# ============================================================================
# Ranking & Buffer Calculations
# ============================================================================

def _lb_family_sp_for(leader: "_Leader", player: str, account: str) -> int:
    """Get family SP for a player from leaderboard."""
    p = _norm_key(player)
    a = _canon_key(account)
    return int(leader["sp_map"].get(p, {}).get(a, 0))


def split_multi_subject_players(players_field: str) -> t.List[str]:
    """Split multi-subject player field by '/' delimiter."""
    parts = [p.strip() for p in str(players_field or "").split("/") if str(p).strip()]
    return parts if parts else []


def _rank_and_buffer_full_leader(
    player: str,
    leader: "_Leader",
    family_eff: t.Dict[str, int]
) -> t.Tuple[t.Optional[int], t.Optional[int], t.Optional[str], t.Optional[int]]:
    """
    Compute rank, buffer, best account, and SP for a player.
    Combines leaderboard + effective family state.
    """
    p = _norm_key(player)
    rows = leader["by_player"].get(p, [])
    
    if not rows and all(v <= 0 for v in family_eff.values()):
        return None, None, None, None
    
    combined: t.Dict[str, int] = {}
    
    # Load leaderboard data
    for r in rows:
        combined[str(r["account"])] = max(int(combined.get(str(r["account"]), 0)), int(r["sp"] or 0))
    
    # Overlay family effective state
    for a_disp, sp in family_eff.items():
        combined[_canon_key(a_disp)] = int(max(0, sp))
    
    # Find family's best account
    fam_best_acct_disp = None
    fam_best_sp = -1
    for a_disp in FAMILY_ACCOUNTS:
        k = _canon_key(a_disp)
        v = int(combined.get(k, 0))
        if v > fam_best_sp:
            fam_best_sp = v
            fam_best_acct_disp = a_disp
    
    # Find best non-family account
    best_nonfamily_sp = 0
    for acc, v in combined.items():
        if acc in _FAMILY_KEYS:
            continue
        best_nonfamily_sp = max(best_nonfamily_sp, int(v))
    
    higher = sum(1 for v in combined.values() if v > fam_best_sp)
    rank = 1 + higher
    buffer = fam_best_sp - best_nonfamily_sp
    
    return rank, buffer, fam_best_acct_disp, fam_best_sp


# ============================================================================
# QP Scoring
# ============================================================================

def _points_for_rank(r: t.Optional[int]) -> int:
    """Get QP points for a rank."""
    return int(QP_MAP.get(int(r), 0)) if r is not None else 0


@log_util.timed_operation("compute_family_qp")
def compute_family_qp_derived_531(
    leader: "_Leader",
    family_sp: t.Dict[str, t.Dict[str, int]],
) -> t.Tuple[int, t.Dict[str, int]]:
    """
    Compute family QP under 5-3-1 contest rules.
    Returns total and per-account breakdown.
    """
    total = 0
    per_acct = {a: 0 for a in FAMILY_ACCOUNTS}
    
    all_players: t.Set[str] = set(leader["sp_map"].keys())
    for a in FAMILY_ACCOUNTS:
        all_players.update(family_sp.get(a, {}).keys())
    
    for pkey in all_players:
        eff = {a: int(family_sp.get(a, {}).get(pkey, 0)) for a in FAMILY_ACCOUNTS}
        r, _b, best_acct, _sp = _rank_and_buffer_full_leader(pkey, leader, eff)
        pts = _points_for_rank(r)
        
        if pts and best_acct:
            total += pts
            per_acct[best_acct] += pts
    
    return int(total), per_acct


# ============================================================================
# Pydantic Models (Requests/Responses)
# ============================================================================

class TradeLine(BaseModel):
    side: t.Literal["GET", "GIVE"]
    players: str
    sp: int = Field(ge=config.MIN_SP_PER_TRADE, le=config.MAX_SP_PER_TRADE_LINE)


class FamilyEvaluateTradeReq(BaseModel):
    prefer_env_defaults: bool = True
    leaderboard_url: t.Optional[str] = None
    player_tags_url: t.Optional[str] = None
    holdings_e31_url: t.Optional[str] = None
    holdings_dc_url: t.Optional[str] = None
    holdings_fe_url: t.Optional[str] = None
    holdings_ud_url: t.Optional[str] = None
    trade_account: t.Literal["Easystreet31", "DusterCrusher", "FinkleIsEinhorn", "UpperDuck"]
    trade: t.List[TradeLine]
    multi_subject_rule: t.Literal["full_each_unique"] = "full_each_unique"
    fragility_mode: t.Literal["trade_delta", "none"] = "trade_delta"
    
    @validator("trade")
    def _non_empty(cls, v):
        if not v:
            raise ValueError("trade must contain at least one line")
        return v


class LeaderboardDeltaReq(BaseModel):
    prefer_env_defaults: bool = True
    leaderboard_today_url: t.Optional[str] = None
    leaderboard_yesterday_url: t.Optional[str] = None
    rivals: t.Optional[t.List[str]] = None
    min_sp_delta: int = Field(default=1, ge=0)


# ============================================================================
# Leaderboard Delta
# ============================================================================

def _delta_rows(
    today: "_Leader",
    yday: "_Leader",
    rivals_canon: t.Set[str],
    min_sp_delta: int
) -> t.List[t.Dict[str, t.Any]]:
    """Compute delta rows between two leaderboards."""
    def rival_sum(leader: _Leader, p: str) -> int:
        sm = leader["sp_map"].get(p, {})
        return int(sum(int(sm.get(rv, 0)) for rv in rivals_canon))
    
    players = set(today["sp_map"].keys()) | set(yday["sp_map"].keys())
    rows: t.List[t.Dict[str, t.Any]] = []
    
    for p in players:
        fam_eff_before = {a: int(yday["sp_map"].get(p, {}).get(_canon_key(a), 0)) for a in FAMILY_ACCOUNTS}
        fam_eff_after = {a: int(today["sp_map"].get(p, {}).get(_canon_key(a), 0)) for a in FAMILY_ACCOUNTS}
        r1, b1, a1, sp1 = _rank_and_buffer_full_leader(p, yday, fam_eff_before)
        r2, b2, a2, sp2 = _rank_and_buffer_full_leader(p, today, fam_eff_after)
        rv1 = rival_sum(yday, p)
        rv2 = rival_sum(today, p)
        
        d_sp = (sp2 or 0) - (sp1 or 0)
        d_buf = (b2 - b1) if (b1 is not None and b2 is not None) else None
        d_rank = (r2 - r1) if (r1 is not None and r2 is not None) else None
        d_rival = rv2 - rv1
        disp = today["display_names"].get(p) or yday["display_names"].get(p) or p
        
        moved = abs(d_sp) >= int(min_sp_delta) or (d_rank not in (None, 0)) or (d_rival != 0)
        if not moved:
            continue
        
        rows.append({
            "player": disp,
            "player_key": p,
            "family_before": {"account": a1, "sp": sp1, "rank": r1, "buffer": b1},
            "family_after": {"account": a2, "sp": sp2, "rank": r2, "buffer": b2},
            "delta_sp": d_sp,
            "delta_rank": d_rank,
            "delta_buffer": d_buf,
            "rivals_sp_before": rv1,
            "rivals_sp_after": rv2,
            "delta_rivals_sp": d_rival
        })
    
    rows.sort(
        key=lambda r: (abs(int(r.get("delta_sp") or 0)), abs(int(r.get("delta_buffer") or 0))),
        reverse=True
    )
    return rows


# ============================================================================
# Trade Evaluation
# ============================================================================

def _detect_tag(player_key: str, tags: t.Dict[str, t.Set[str]]) -> t.Optional[str]:
    """Detect player tag."""
    for k in config.PLAYER_TAG_CATEGORIES:
        if player_key in tags.get(k, set()):
            return k
    return None


@log_util.timed_operation("evaluate_trade")
def evaluate_trade_internal(
    req: FamilyEvaluateTradeReq,
    leader: "_Leader",
    tags: t.Dict[str, t.Set[str]],
    holds: t.Dict[str, t.Dict[str, int]]
) -> t.Dict[str, t.Any]:
    """Core trade evaluation logic."""
    
    # Family SP before (effective = max(leaderboard, holdings))
    fam_before: t.Dict[str, t.Dict[str, int]] = {a: {} for a in FAMILY_ACCOUNTS}
    all_players: t.Set[str] = set(leader["sp_map"].keys())
    for a in FAMILY_ACCOUNTS:
        all_players.update(_norm_key(k) for k in holds.get(a, {}).keys())
    
    for p in all_players:
        for a in FAMILY_ACCOUNTS:
            lb = _lb_family_sp_for(leader, p, a)
            hold = int(holds.get(a, {}).get(p, 0))
            fam_before[a][p] = max(lb, hold)
    
    # Accumulate GET and GIVE by player
    get_add: t.Dict[str, int] = defaultdict(int)
    give_subtract: t.Dict[str, int] = defaultdict(int)
    
    for line in req.trade:
        subjects = (
            split_multi_subject_players(line.players)
            if req.multi_subject_rule == "full_each_unique"
            else [line.players]
        )
        for subj in subjects:
            pkey = _norm_key(subj)
            if line.side == "GET":
                get_add[pkey] += int(line.sp)
            else:
                give_subtract[pkey] += int(line.sp)
    
    # Route GETs
    allocation_plan: t.List[t.Dict[str, t.Any]] = []
    routed_to: t.Dict[str, t.Tuple[str, int]] = {}
    
    for pkey, add_sp in get_add.items():
        tag = _detect_tag(pkey, tags)
        order = config.ROUTING_PRIORITY.get(tag, config.DEFAULT_ROUTING_ORDER)
        
        to = order[0]  # Simplified routing for now
        routed_to[pkey] = (to, add_sp)
        
        allocation_plan.append({
            "type": "GET",
            "to": to,
            "players": [leader["display_names"].get(pkey, pkey)],
            "sp": int(add_sp),
            "tag": tag,
        })
    
    # Apply deltas
    fam_after: t.Dict[str, t.Dict[str, int]] = {
        a: dict(spmap) for a, spmap in fam_before.items()
    }
    
    for pkey, (to, add_sp) in routed_to.items():
        fam_after[to][pkey] = int(fam_after[to].get(pkey, 0)) + int(add_sp)
    
    for pkey, sub_sp in give_subtract.items():
        fam_after[req.trade_account][pkey] = max(
            0, int(fam_after[req.trade_account].get(pkey, 0)) - int(sub_sp)
        )
    
    # Compute changes
    touched = set(get_add.keys()) | set(give_subtract.keys())
    player_changes: t.List[t.Dict[str, t.Any]] = []
    
    for pkey in sorted(touched):
        disp = leader["display_names"].get(pkey, pkey)
        per_before = {a: int(fam_before[a].get(pkey, 0)) for a in FAMILY_ACCOUNTS}
        per_after = {a: int(fam_after[a].get(pkey, 0)) for a in FAMILY_ACCOUNTS}
        sp_before = sum(per_before.values())
        sp_after = sum(per_after.values())
        delta_sp = sp_after - sp_before
        
        r1, b1, _a1, _sp1 = _rank_and_buffer_full_leader(pkey, leader, per_before)
        r2, b2, _a2, _sp2 = _rank_and_buffer_full_leader(pkey, leader, per_after)
        
        qp_before = _points_for_rank(r1)
        qp_after = _points_for_rank(r2)
        d_qp = qp_after - qp_before
        
        change = {
            "player": disp,
            "sp_before": sp_before,
            "sp_after": sp_after,
            "delta_sp": delta_sp,
            "per_account_sp_before": per_before,
            "per_account_sp_after": per_after,
            "per_account_sp_delta": {a: per_after[a] - per_before[a] for a in FAMILY_ACCOUNTS},
            "best_rank_before": r1,
            "best_rank_after": r2,
            "buffer_before": b1,
            "buffer_after": b2,
            "delta_buffer": (b2 - b1) if (b1 is not None and b2 is not None) else None,
            "qp_before": qp_before,
            "qp_after": qp_after,
            "delta_qp": d_qp,
        }
        player_changes.append(change)
    
    # Totals
    total_sp = sum(int(p["delta_sp"] or 0) for p in player_changes)
    total_buf = sum(int(p["delta_buffer"] or 0) for p in player_changes if p["delta_buffer"] is not None)
    total_qp = int(sum(int(p["delta_qp"] or 0) for p in player_changes))
    
    verdict = "APPROVE" if total_qp > 0 else ("CAUTION" if total_qp == 0 else "DECLINE")
    
    # Family QP snapshots
    fam_qp_before, _ = compute_family_qp_derived_531(leader, fam_before)
    fam_qp_after, _ = compute_family_qp_derived_531(leader, fam_after)
    
    return {
        "ok": True,
        "allocation_plan": allocation_plan,
        "player_changes": player_changes,
        "total_changes": {"delta_sp": total_sp, "delta_buffer": total_buf, "delta_qp": total_qp},
        "verdict": verdict,
        "ownership_warnings": [],
        "family_qp_before": int(fam_qp_before),
        "family_qp_after": int(fam_qp_after),
    }
