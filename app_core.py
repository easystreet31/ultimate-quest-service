import io
import os
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
    version=os.getenv("APP_VERSION", "4.12.6"),
)

# --------------------------------------------------------------------------------------
# Family accounts (display names) and canonicalization
# --------------------------------------------------------------------------------------

FAMILY_ACCOUNTS: t.List[str] = [
    "Easystreet31",
    "DusterCrusher",
    "FinkleIsEinhorn",
    "UpperDuck",
]

def _canon_key(s: t.Any) -> str:
    """
    Canonicalize account identifiers so leaderboard usernames like
    'Finkle Is Einhorn' match 'FinkleIsEinhorn':
      - lowercase
      - remove all non-alphanumeric characters
    """
    raw = "".join(ch for ch in str(s or "") if ch.isalnum())
    return raw.lower()

_FAMILY_CANON_TO_DISPLAY: t.Dict[str, str] = { _canon_key(a): a for a in FAMILY_ACCOUNTS }
_FAMILY_KEYS: t.Set[str] = set(_FAMILY_CANON_TO_DISPLAY.keys())

# --------------------------------------------------------------------------------------
# Rivals (from env)
# --------------------------------------------------------------------------------------

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
    raw = _http_get_bytes(url)
    with pd.ExcelFile(io.BytesIO(raw)) as xf:
        sheets = {}
        for name in xf.sheet_names:
            df = xf.parse(name)
            # keep original header spellings; normalize later
            df.columns = [str(c).strip() for c in df.columns]
            sheets[name] = df
        return sheets

def _norm_name(s: t.Any) -> str:
    return str(s or "").strip()

def _norm_key(s: t.Any) -> str:
    return _norm_name(s).lower()

def _detect_col(df: pd.DataFrame, candidates: t.Iterable[str]) -> str:
    """Exact match (case-insensitive) for common column names."""
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for want in candidates:
        lw = want.lower()
        if lw in lower_map:
            return lower_map[lw]
    # heuristics for 'user'/'account' family
    for c in df.columns:
        cl = str(c).lower().strip()
        if any(tok in cl for tok in ("username", "user_name")) and "user" in [w.lower() for w in candidates]:
            return c
        if any(tok in cl for tok in ("handle",)) and "user" in [w.lower() for w in candidates]:
            return c
        if any(tok in cl for tok in ("account_name", "acct", "acct_name")) and "account" in [w.lower() for w in candidates]:
            return c
    raise KeyError(f"Missing required column (tried {list(candidates)}) in columns: {list(df.columns)}")

def _detect_col_qp(df: pd.DataFrame) -> t.Optional[str]:
    """
    Fuzzy detector for QP column. Accepts: 'qp', 'QP', 'quest', 'quest_points',
    'quest points', 'questpoints', 'quest total', etc.
    """
    # First try the exact names
    for base in ("qp", "quest", "quest_points"):
        try:
            return _detect_col(df, [base])
        except KeyError:
            pass
    # Fuzzy contains
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl == "qp":
            return c
        if "quest" in cl and ("point" in cl or "total" in cl or "qp" in cl):
            return c
        if "qp" in cl:
            return c
    return None

# --------------------------------------------------------------------------------------
# Leaderboard normalization (store *canonical* account keys)
# --------------------------------------------------------------------------------------

class _Row(t.TypedDict, total=False):
    account: str   # canonical key (alnum-only, lowercase)
    sp: int
    qp: int
    rank: t.Optional[int]

class _Leader(t.TypedDict):
    by_player: t.Dict[str, t.List[_Row]]
    sp_map: t.Dict[str, t.Dict[str, int]]  # {player_key: {account_canon: sp}}

def normalize_leaderboard(sheets: t.Dict[str, pd.DataFrame]) -> _Leader:
    name = next(iter(sheets))
    df = sheets[name].copy()

    col_player  = _detect_col(df, ["player", "players", "subject", "name"])
    col_account = _detect_col(df, ["account", "username", "user", "owner", "handle"])
    col_sp      = _detect_col(df, ["sp", "score", "points"])
    col_qp      = _detect_col_qp(df)

    col_rank = None
    for cand in ("rank", "position"):
        try:
            col_rank = _detect_col(df, [cand]); break
        except KeyError:
            continue

    df[col_player]  = df[col_player].map(_norm_name)
    df[col_account] = df[col_account].map(_norm_name)
    df[col_sp]      = pd.to_numeric(df[col_sp], errors="coerce").fillna(0).astype(int)
    if col_qp:
        df[col_qp]   = pd.to_numeric(df[col_qp], errors="coerce").fillna(0).astype(int)
    if col_rank:
        df[col_rank] = pd.to_numeric(df[col_rank], errors="coerce").fillna(0).astype(int)

    by_player: t.Dict[str, t.List[_Row]] = defaultdict(list)
    sp_map: t.Dict[str, t.Dict[str, int]] = defaultdict(dict)

    for _, row in df.iterrows():
        p = _norm_key(row[col_player])
        a_canon = _canon_key(row[col_account])
        sp = int(row[col_sp]) if pd.notna(row[col_sp]) else 0
        qp = int(row[col_qp]) if col_qp and pd.notna(row[col_qp]) else 0
        rk = int(row[col_rank]) if col_rank and pd.notna(row[col_rank]) else None

        by_player[p].append(_Row(account=a_canon, sp=sp, qp=qp, rank=rk))
        # Record the max SP seen for (player, account)
        prev = int(sp_map[p].get(a_canon, 0))
        sp_map[p][a_canon] = max(prev, sp)

    return t.cast(_Leader, {"by_player": dict(by_player), "sp_map": dict(sp_map)})

# --------------------------------------------------------------------------------------
# Holdings & tags loaders
# --------------------------------------------------------------------------------------

def _frame_to_holdings(df: pd.DataFrame) -> t.Dict[str, int]:
    col_player = _detect_col(df, ["player", "players", "subject", "name"])
    col_sp = None
    for c in ("sp", "score", "points", "count", "qty", "quantity"):
        try:
            col_sp = _detect_col(df, [c]); break
        except KeyError:
            continue
    if not col_sp:
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
            df = xf.parse(xf.sheet_names[0])
            out[acct] = _frame_to_holdings(df)
    return out

def _load_player_tags(prefer_env_defaults: bool, player_tags_url: t.Optional[str]) -> t.Dict[str, t.Set[str]]:
    url = _pick_url(player_tags_url, "player_tags", prefer_env_defaults)
    raw = _http_get_bytes(url)
    tags: t.Dict[str, t.Set[str]] = {"LEGENDS": set(), "ANA": set(), "DAL": set(), "LAK": set(), "PIT": set()}
    with pd.ExcelFile(io.BytesIO(raw)) as xf:
        for tab, key in [("Legends", "LEGENDS"), ("ANA", "ANA"), ("DAL", "DAL"), ("LAK", "LAK"), ("PIT", "PIT")]:
            if tab not in xf.sheet_names:
                continue
            df = xf.parse(tab)
            if df.empty:
                continue
            col = df.columns[0]
            vals = [_norm_key(v) for v in list(df[col].astype(str)) if str(v).strip()]
            tags[key] |= set(vals)
    return tags

# --------------------------------------------------------------------------------------
# Rank/buffer/QP helpers  (use canonical keys for accounts)
# --------------------------------------------------------------------------------------

def _lb_family_sp_for(leader: "_Leader", player: str, account: str) -> int:
    p = _norm_key(player)
    a = _canon_key(account)
    return int(leader["sp_map"].get(p, {}).get(a, 0))

def split_multi_subject_players(players_field: str) -> t.List[str]:
    parts = [p.strip() for p in str(players_field or "").split("/") if str(p).strip()]
    return parts if parts else []

def _rank_and_buffer_full_leader(
    player: str, leader: "_Leader", family_eff: t.Dict[str, int]
) -> t.Tuple[t.Optional[int], t.Optional[int], t.Optional[str], t.Optional[int]]:
    """
    Compute best family rank among ALL accounts on leaderboard + the buffer vs best non-family.
    Accounts in 'leader' are canonical; we compare against canonical family keys.
    """
    p = _norm_key(player)
    rows = leader["by_player"].get(p, [])
    if not rows and all(v <= 0 for v in family_eff.values()):
        return None, None, None, None

    combined: t.Dict[str, int] = {}
    for r in rows:
        combined[str(r["account"])] = max(int(combined.get(str(r["account"]), 0)), int(r["sp"] or 0))
    for a_disp, sp in family_eff.items():
        combined[_canon_key(a_disp)] = int(max(0, sp))

    # Determine best family & best rival
    fam_best_acct_disp = None
    fam_best_sp = -1
    for a_disp in FAMILY_ACCOUNTS:
        k = _canon_key(a_disp)
        v = int(combined.get(k, 0))
        if v > fam_best_sp or (v == fam_best_sp and fam_best_acct_disp is not None and FAMILY_ACCOUNTS.index(a_disp) < FAMILY_ACCOUNTS.index(fam_best_acct_disp)):
            fam_best_sp = v
            fam_best_acct_disp = a_disp

    best_nonfamily_sp = 0
    for acc, v in combined.items():
        if acc in _FAMILY_KEYS:
            continue
        best_nonfamily_sp = max(best_nonfamily_sp, int(v))

    higher = sum(1 for v in combined.values() if v > fam_best_sp)
    rank = 1 + higher
    buffer = fam_best_sp - best_nonfamily_sp
    return rank, buffer, fam_best_acct_disp, fam_best_sp

def _rank_context_smallset(player: str, leader: "_Leader", eff_map: t.Dict[str, t.Dict[str, int]]) -> t.Dict[str, t.Any]:
    fam_eff = {a: int(eff_map.get(a, {}).get(player, 0)) for a in FAMILY_ACCOUNTS}
    r, _buf, _acct, _sp = _rank_and_buffer_full_leader(player, leader, fam_eff)
    qp = 1 if (r is not None and r == 1) else 0
    return {"family_qp_player": qp, "rank": r}

def compute_family_qp(
    leader: "_Leader", accounts: t.Dict[str, t.Dict[str, int]]
) -> t.Tuple[int, t.Dict[str, int], t.Dict[str, t.Any]]:
    """
    Hybrid QP semantics:
      1) Try leaderboard-QP sum (sum 'qp' for rows whose account belongs to the family,
         after canonicalizing account names).
      2) If that total is 0, fall back to derived QP = number of players where the family
         holds Rank-1 (credit the point to the best family account).
    Returns: (family_qp_total, per_account_qp_map, details)
    """
    # (1) Leaderboard-QP sum
    lb_total = 0
    lb_per_acct = {a: 0 for a in FAMILY_ACCOUNTS}
    for rows in leader["by_player"].values():
        for r in rows:
            acc_canon = str(r.get("account") or "")
            qp = int(r.get("qp") or 0)
            if acc_canon in _FAMILY_KEYS:
                lb_total += qp
                disp = _FAMILY_CANON_TO_DISPLAY.get(acc_canon)
                if disp:
                    lb_per_acct[disp] += qp

    if lb_total > 0:
        details: t.Dict[str, t.Any] = {
            "source": "leaderboard_qp_sum",
            "lb_qp_sum": int(lb_total),
            "derived_rank1_count": None,
            "per_account_qp": lb_per_acct,
        }
        return int(lb_total), lb_per_acct, details

    # (2) Derived QP: count family Rank-1s
    derived_total = 0
    derived_per_acct = {a: 0 for a in FAMILY_ACCOUNTS}

    # Build 'effective' family SP map for each player (max of LB vs holdings)
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
            derived_total += 1
            derived_per_acct[best_a] += 1

    details = {
        "source": "derived_rank1_count",
        "lb_qp_sum": int(lb_total),
        "derived_rank1_count": int(derived_total),
        "per_account_qp": derived_per_acct,
    }
    return int(derived_total), derived_per_acct, details

# --------------------------------------------------------------------------------------
# (The rest of your endpoints live in main.py; we only expose /defaults + helpers here)
# --------------------------------------------------------------------------------------
