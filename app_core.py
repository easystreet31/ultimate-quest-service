import io
import os
import math
import typing as t
from collections import defaultdict

import pandas as pd
import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator

# --------------------------------------------------------------------------------------
# FastAPI app (imported by main.py as "core_app")
# --------------------------------------------------------------------------------------

app = FastAPI(
    title="Ultimate Quest Service (Small-Payload API)",
    version=os.getenv("APP_VERSION", "4.12.2"),
)

# --------------------------------------------------------------------------------------
# Constants / family
# --------------------------------------------------------------------------------------

FAMILY_ACCOUNTS: t.List[str] = [
    "Easystreet31",
    "DusterCrusher",
    "FinkleIsEinhorn",
    "UpperDuck",
]

SYNDICATE: t.Set[str] = set(
    (os.getenv("DEFAULT_TARGET_RIVALS") or "chfkyle,VjV5,FireRanger,Tfunite,Ovi8")
    .lower()
    .split(",")
)

# --------------------------------------------------------------------------------------
# Models used by main.py (do not remove/rename)
# --------------------------------------------------------------------------------------

class TradeLine(BaseModel):
    side: t.Literal["GET", "GIVE"]
    players: str  # one player or "A/B/C" multi-subject
    sp: int = Field(ge=1)

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

# --------------------------------------------------------------------------------------
# /defaults  (simple env projection used by clients)
# --------------------------------------------------------------------------------------

def _env(k: str, default: t.Optional[str] = None) -> t.Optional[str]:
    v = os.getenv(k)
    return v if v not in (None, "", "REPLACE-HOLDINGS-UD", "REPLACE-COLLECTION-UD") else default

@app.get("/defaults")
def defaults():
    return {
        "leaderboard": _env("DEFAULT_LEADERBOARD_URL"),
        "leaderboard_yday": _env("DEFAULT_LEADERBOARD_YDAY_URL"),

        "holdings_e31": _env("DEFAULT_HOLDINGS_E31_URL"),
        "holdings_dc":  _env("DEFAULT_HOLDINGS_DC_URL"),
        "holdings_fe":  _env("DEFAULT_HOLDINGS_FE_URL"),
        "holdings_ud":  _env("DEFAULT_HOLDINGS_UD_URL"),

        "collection_e31": _env("DEFAULT_COLLECTION_E31_URL"),
        "collection_dc":  _env("DEFAULT_COLLECTION_DC_URL"),
        "collection_fe":  _env("DEFAULT_COLLECTION_FE_URL"),
        "collection_ud":  _env("DEFAULT_COLLECTION_UD_URL"),

        "pool_collection": _env("DEFAULT_POOL_COLLECTION_URL"),
        "player_tags": _env("PLAYER_TAGS_URL"),

        "rivals": list(SYNDICATE),
        "defend_buffer_all": int(os.getenv("DEFAULT_DEFEND_BUFFER_ALL", "15")),
        "fragility_default": os.getenv("TRADE_FRAGILITY_DEFAULT", "trade_delta"),
        "force_family_urls": os.getenv("FORCE_FAMILY_URLS", "true").lower() in ("1","true","yes","y","on"),
    }

# --------------------------------------------------------------------------------------
# Helpers the main app imports
# --------------------------------------------------------------------------------------

def _pick_url(explicit: t.Optional[str], kind: str, prefer_env_defaults: bool) -> str:
    """
    Choose an explicit URL if provided; otherwise fall back to env defaults by kind.
    """
    if explicit and explicit.strip():
        return explicit.strip()

    env_map = {
        "leaderboard": "DEFAULT_LEADERBOARD_URL",
        "leaderboard_yday": "DEFAULT_LEADERBOARD_YDAY_URL",
        "holdings_e31": "DEFAULT_HOLDINGS_E31_URL",
        "holdings_dc":  "DEFAULT_HOLDINGS_DC_URL",
        "holdings_fe":  "DEFAULT_HOLDINGS_FE_URL",
        "holdings_ud":  "DEFAULT_HOLDINGS_UD_URL",
        "collection_e31": "DEFAULT_COLLECTION_E31_URL",
        "collection_dc":  "DEFAULT_COLLECTION_DC_URL",
        "collection_fe":  "DEFAULT_COLLECTION_FE_URL",
        "collection_ud":  "DEFAULT_COLLECTION_UD_URL",
        "pool_collection": "DEFAULT_POOL_COLLECTION_URL",
        "player_tags": "PLAYER_TAGS_URL",
    }
    if not prefer_env_defaults:
        raise ValueError(f"No explicit URL for '{kind}' and prefer_env_defaults is False")
    k = env_map.get(kind)
    if not k:
        raise ValueError(f"Unknown url kind: {kind}")
    v = os.getenv(k)
    if not v:
        raise ValueError(f"Missing environment default for {kind}: {k}")
    return v

def _http_get_bytes(url: str) -> bytes:
    headers = {
        "User-Agent": "ultimate-quest-service/1.0",
        "Accept": "*/*",
    }
    r = requests.get(url, headers=headers, timeout=45)
    r.raise_for_status()
    return r.content

def fetch_xlsx(url: str) -> t.Dict[str, pd.DataFrame]:
    """
    Download a Google Sheets 'export?format=xlsx' file and return a dict of sheetname->DataFrame.
    """
    raw = _http_get_bytes(url)
    with pd.ExcelFile(io.BytesIO(raw)) as xf:
        sheets = {}
        for name in xf.sheet_names:
            df = xf.parse(name)
            # Normalize column names to strings (keep original case for content)
            df.columns = [str(c).strip() for c in df.columns]
            sheets[name] = df
        return sheets

def _norm_name(s: t.Any) -> str:
    return str(s or "").strip()

def _norm_key(s: t.Any) -> str:
    return _norm_name(s).lower()

def _detect_col(df: pd.DataFrame, candidates: t.Iterable[str]) -> str:
    """
    Find the first matching column from candidates (case-insensitive).
    Raises KeyError if none found.
    """
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for want in candidates:
        lw = want.lower()
        if lw in lower_map:
            return lower_map[lw]
    # try prefix/contains heuristics for 'user'/'acct' variants (e.g., 'username')
    for c in df.columns:
        cl = str(c).lower()
        if any(tok in cl for tok in ("username", "user_name")) and "user" in [w.lower() for w in candidates]:
            return c
        if any(tok in cl for tok in ("account_name", "acct", "acct_name")) and "account" in [w.lower() for w in candidates]:
            return c
    raise KeyError(f"Missing required column (tried {list(candidates)}) in columns: {list(df.columns)}")

# --------------------------------------------------------------------------------------
# Leaderboard normalization
# --------------------------------------------------------------------------------------

class _Leader(t.TypedDict):
    # per-player -> list of (account, sp, qp, rank) rows from the sheet
    by_player: t.Dict[str, t.List[t.Dict[str, t.Any]]]
    # quick lookup: per-player -> {account_lower: sp}
    sp_map: t.Dict[str, t.Dict[str, int]]

def normalize_leaderboard(sheets: t.Dict[str, pd.DataFrame]) -> _Leader:
    """
    Accepts the dict returned by fetch_xlsx and builds a structure suitable
    for rank/buffer/QP computations. We are defensive on column names.
    """
    # Pick the first sheet (your export usually has a single sheet named "subject_leaderboards.csv")
    name = next(iter(sheets))
    df = sheets[name].copy()

    # Detect columns (accept common aliases)
    col_player = _detect_col(df, ["player", "players", "subject", "name"])
    col_account = _detect_col(df, ["account", "username", "user", "owner", "handle"])
    col_sp = _detect_col(df, ["sp", "score", "points"])
    # Optional columns
    col_qp = None
    for cand in ("qp", "quest", "quest_points"):
        try:
            col_qp = _detect_col(df, [cand])
            break
        except KeyError:
            continue
    col_rank = None
    for cand in ("rank", "position"):
        try:
            col_rank = _detect_col(df, [cand])
            break
        except KeyError:
            continue

    # Clean & coerce
    df[col_player] = df[col_player].map(_norm_name)
    df[col_account] = df[col_account].map(_norm_name)
    df[col_sp] = pd.to_numeric(df[col_sp], errors="coerce").fillna(0).astype(int)
    if col_qp:
        df[col_qp] = pd.to_numeric(df[col_qp], errors="coerce").fillna(0).astype(int)
    if col_rank:
        df[col_rank] = pd.to_numeric(df[col_rank], errors="coerce").fillna(0).astype(int)

    by_player: t.Dict[str, t.List[t.Dict[str, t.Any]]] = defaultdict(list)
    sp_map: t.Dict[str, t.Dict[str, int]] = defaultdict(dict)

    for _, row in df.iterrows():
        p = _norm_key(row[col_player])
        a = _norm_key(row[col_account])
        sp = int(row[col_sp]) if pd.notna(row[col_sp]) else 0
        qp = int(row[col_qp]) if col_qp and pd.notna(row[col_qp]) else 0
        rk = int(row[col_rank]) if col_rank and pd.notna(row[col_rank]) else None
        by_player[p].append({"account": a, "sp": sp, "qp": qp, "rank": rk})
        # Record the max SP seen for (player, account)
        prev = int(sp_map[p].get(a, 0))
        sp_map[p][a] = max(prev, sp)

    return t.cast(_Leader, {"by_player": dict(by_player), "sp_map": dict(sp_map)})

# --------------------------------------------------------------------------------------
# Holdings & tags loaders
# --------------------------------------------------------------------------------------

def _frame_to_holdings(df: pd.DataFrame) -> t.Dict[str, int]:
    """
    Accepts a sheet with (player, sp or count) and returns {player_name: sp}.
    We'll prefer 'sp' if present; otherwise use 'count' as SP.
    """
    # Try to detect the two main columns
    col_player = _detect_col(df, ["player", "players", "subject", "name"])
    col_sp = None
    for c in ("sp", "score", "points", "count", "qty", "quantity"):
        try:
            col_sp = _detect_col(df, [c])
            break
        except KeyError:
            continue
    if not col_sp:
        # Fallback: any numeric column
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return {}
        col_sp = numeric_cols[0]

    df = df[[col_player, col_sp]].copy()
    df[col_player] = df[col_player].map(_norm_name)
    df[col_sp] = pd.to_numeric(df[col_sp], errors="coerce").fillna(0).astype(int)
    grp = df.groupby(col_player, dropna=False)[col_sp].sum().reset_index()
    return { _norm_name(r[col_player]): int(r[col_sp]) for _, r in grp.iterrows() }

def holdings_from_urls(
    holdings_e31_url: t.Optional[str],
    holdings_dc_url: t.Optional[str],
    holdings_fe_url: t.Optional[str],
    prefer_env_defaults: bool,
    holdings_ud_url: t.Optional[str],
) -> t.Dict[str, t.Dict[str, int]]:
    """
    Load four account holdings from explicit urls or env defaults.
    Returns: {account: {player_name: sp}}
    """
    urls = {
        "Easystreet31": _pick_url(holdings_e31_url, "holdings_e31", prefer_env_defaults),
        "DusterCrusher": _pick_url(holdings_dc_url,  "holdings_dc",  prefer_env_defaults),
        "FinkleIsEinhorn": _pick_url(holdings_fe_url,  "holdings_fe",  prefer_env_defaults),
        "UpperDuck": _pick_url(holdings_ud_url,  "holdings_ud",  prefer_env_defaults),
    }
    out: t.Dict[str, t.Dict[str, int]] = {a: {} for a in FAMILY_ACCOUNTS}
    for acct, url in urls.items():
        raw = _http_get_bytes(url)
        with pd.ExcelFile(io.BytesIO(raw)) as xf:
            # Use the first sheet by convention
            df = xf.parse(xf.sheet_names[0])
            out[acct] = _frame_to_holdings(df)
    return out

def _load_player_tags(prefer_env_defaults: bool, player_tags_url: t.Optional[str]) -> t.Dict[str, t.Set[str]]:
    """
    Load tags workbook (tabs: Legends, ANA, DAL, LAK, PIT).
    Returns lower-cased player name sets per tag.
    """
    url = _pick_url(player_tags_url, "player_tags", prefer_env_defaults)
    raw = _http_get_bytes(url)
    tags: t.Dict[str, t.Set[str]] = {"LEGENDS": set(), "ANA": set(), "DAL": set(), "LAK": set(), "PIT": set()}
    with pd.ExcelFile(io.BytesIO(raw)) as xf:
        for tab, key in [
            ("Legends", "LEGENDS"),
            ("ANA", "ANA"), ("DAL", "DAL"), ("LAK", "LAK"), ("PIT", "PIT")
        ]:
            if tab not in xf.sheet_names:
                continue
            df = xf.parse(tab)
            if df.empty:
                continue
            # First non-empty column
            col = df.columns[0]
            vals = [ _norm_key(v) for v in list(df[col].astype(str)) if str(v).strip() ]
            tags[key] |= set(vals)
    return tags

# --------------------------------------------------------------------------------------
# Rank/buffer/QP helpers
# --------------------------------------------------------------------------------------

def _lb_family_sp_for(leader: "_Leader", player: str, account: str) -> int:
    """SP of (player, account) from the leaderboard rows (max if dup)."""
    p = _norm_key(player)
    a = _norm_key(account)
    return int(leader["sp_map"].get(p, {}).get(a, 0))

def split_multi_subject_players(players_field: str) -> t.List[str]:
    """Split 'A/B/C' or single name into a clean list of player names (original casing kept)."""
    parts = [p.strip() for p in str(players_field or "").split("/") if str(p).strip()]
    return parts if parts else []

def _rank_and_buffer_full_leader(player: str, leader: "_Leader", family_eff: t.Dict[str, int]) -> t.Tuple[t.Optional[int], t.Optional[int], t.Optional[str], t.Optional[int]]:
    """
    Compute best family rank among ALL accounts on leaderboard + the buffer vs best non-family.
    - rank: 1 = top; None if player absent.
    - buffer: best_family_sp - best_nonfamily_sp (can be negative).
    - best_family_account: which family account holds best_family_sp (by value, tie-broken by FAMILY_ACCOUNTS order).
    - best_family_sp: returned as the 4th value for convenience.
    """
    p = _norm_key(player)
    rows = leader["by_player"].get(p, [])
    if not rows and all(v <= 0 for v in family_eff.values()):
        return None, None, None, None

    # Build combined map of account->sp (family overlay on top of leaderboard rows)
    combined: t.Dict[str, int] = {}
    for r in rows:
        combined[r["account"]] = max(int(combined.get(r["account"], 0)), int(r["sp"] or 0))
    for a, sp in family_eff.items():
        combined[_norm_key(a)] = int(max(0, sp))

    # Determine best family & best rival
    fam_best_acct = None
    fam_best_sp = -1
    for a in FAMILY_ACCOUNTS:
        k = _norm_key(a)
        v = int(combined.get(k, 0))
        if v > fam_best_sp or (v == fam_best_sp and fam_best_acct is not None and FAMILY_ACCOUNTS.index(a) < FAMILY_ACCOUNTS.index(fam_best_acct)):
            fam_best_sp = v
            fam_best_acct = a

    best_nonfamily_sp = 0
    fam_keys = { _norm_key(a) for a in FAMILY_ACCOUNTS }
    for acc, v in combined.items():
        if acc in fam_keys:
            continue
        best_nonfamily_sp = max(best_nonfamily_sp, int(v))

    # Rank = 1 + number of accounts strictly above best_family_sp
    higher = sum(1 for v in combined.values() if v > fam_best_sp)
    rank = 1 + higher
    buffer = fam_best_sp - best_nonfamily_sp
    return rank, buffer, fam_best_acct, fam_best_sp

def _rank_context_smallset(player: str, leader: "_Leader", eff_map: t.Dict[str, t.Dict[str, int]]) -> t.Dict[str, t.Any]:
    """
    Minimal per-player context used by main.py:
    - family_qp_player: 1 if family holds rank 1 after considering eff_map, else 0
    """
    fam_eff = {a: int(eff_map.get(a, {}).get(player, 0)) for a in FAMILY_ACCOUNTS}
    r, _buf, _acct, _sp = _rank_and_buffer_full_leader(player, leader, fam_eff)
    qp = 1 if (r is not None and r == 1) else 0
    return {
        "family_qp_player": qp,
        "rank": r,
    }

def compute_family_qp(leader: "_Leader", accounts: t.Dict[str, t.Dict[str, int]]) -> t.Tuple[int, t.Dict[str, int], t.Dict[str, t.Any]]:
    """
    Very small, deterministic QP definition:
      For each player, if family's best rank is 1 -> +1 QP; else 0.
      Attribute the point to the best family account (tie-broken by FAMILY_ACCOUNTS order).
    Returns: (family_qp_total, per_account_qp_map, details)
    """
    total_qp = 0
    per_acct_qp = {a: 0 for a in FAMILY_ACCOUNTS}
    details: t.Dict[str, t.Any] = {}

    # Build 'effective' family SP map for each player
    # effective SP = max(leaderboard SP, holdings SP)
    all_players: t.Set[str] = set(leader["by_player"].keys())
    for a in FAMILY_ACCOUNTS:
        all_players.update(_norm_key(p) for p in accounts.get(a, {}).keys())

    for p in all_players:
        fam_eff = {}
        for a in FAMILY_ACCOUNTS:
            lb = _lb_family_sp_for(leader, p, a)
            hold = int(accounts.get(a, {}).get(p, 0))
            fam_eff[a] = max(lb, hold)

        r, _buf, best_a, _sp = _rank_and_buffer_full_leader(p, leader, fam_eff)
        if r is not None and r == 1 and best_a:
            total_qp += 1
            per_acct_qp[best_a] += 1

    details["per_account_qp"] = per_acct_qp
    return int(total_qp), per_acct_qp, details

# --------------------------------------------------------------------------------------
# (The rest of your endpoints stay in main.py; we only expose /defaults here)
# --------------------------------------------------------------------------------------
