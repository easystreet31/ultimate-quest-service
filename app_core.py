import io
import os
import typing as t
from collections import defaultdict
from datetime import datetime

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field, validator

# ======================================================================================
# FastAPI app metadata
# ======================================================================================

app = FastAPI(
    title="Ultimate Quest Service (Small-Payload API)",
    version=os.getenv("APP_VERSION", "4.12.14"),
)

# ======================================================================================
# Family accounts, rivals, helpers
# ======================================================================================

FAMILY_ACCOUNTS: t.List[str] = [
    "Easystreet31",
    "DusterCrusher",
    "FinkleIsEinhorn",
    "UpperDuck",
]

# Contest QP scoring: Rank 1=5, Rank 2=3, Rank 3=1, else 0
QP_MAP = {1: 5, 2: 3, 3: 1}

def _strip_user_suffix(s: t.Any) -> str:
    s = str(s or "")
    s = s.split("\n", 1)[0]
    s = s.split("(", 1)[0]
    return s.strip()

def _canon_key(s: t.Any) -> str:
    raw = "".join(ch for ch in str(s or "") if ch.isalnum())
    return raw.lower()

def _norm_name(s: t.Any) -> str:
    return str(s or "").strip()

def _norm_key(s: t.Any) -> str:
    return _norm_name(s).lower()

_FAMILY_CANON_TO_DISPLAY: t.Dict[str, str] = { _canon_key(a): a for a in FAMILY_ACCOUNTS }
_FAMILY_KEYS: t.Set[str] = set(_FAMILY_CANON_TO_DISPLAY.keys())

SYNDICATE: t.Set[str] = set(
    (os.getenv("DEFAULT_TARGET_RIVALS") or "chfkyle,VjV5,FireRanger,Tfunite,Ovi8")
    .lower()
    .split(",")
)

# ======================================================================================
# Models (Evaluate + Delta)
# ======================================================================================

class TradeLine(BaseModel):
    side: t.Literal["GET", "GIVE"]
    players: str
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

class LeaderboardDeltaReq(BaseModel):
    prefer_env_defaults: bool = True
    leaderboard_today_url: t.Optional[str] = None
    leaderboard_yesterday_url: t.Optional[str] = None
    rivals: t.Optional[t.List[str]] = None
    min_sp_delta: int = Field(default=1, ge=0)

# ======================================================================================
# /info
# ======================================================================================

@app.get("/info")
def info():
    return {
        "ok": True,
        "title": app.title,
        "version": app.version,
        "default_response_class": "SafeJSONResponse",
    }

# ======================================================================================
# Defaults / HTTP / XLSX helpers
# ======================================================================================

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
    headers = {"User-Agent": "ultimate-quest-service/1.0", "Accept": "*/*"}
    r = requests.get(url, headers=headers, timeout=45)
    r.raise_for_status()
    return r.content

def fetch_xlsx(url: str) -> t.Dict[str, pd.DataFrame]:
    raw = _http_get_bytes(url)
    with pd.ExcelFile(io.BytesIO(raw)) as xf:
        sheets = {}
        for name in xf.sheet_names:
            df = xf.parse(name)
            df.columns = [str(c).strip() for c in df.columns]
            sheets[name] = df
        return sheets

def _detect_col(df: pd.DataFrame, candidates: t.Iterable[str]) -> str:
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for want in candidates:
        lw = want.lower()
        if lw in lower_map:
            return lower_map[lw]
    for c in df.columns:
        cl = str(c).lower().strip()
        if any(tok in cl for tok in ("username","user_name","handle")) and "user" in [w.lower() for w in candidates]:
            return c
        if any(tok in cl for tok in ("account_name","acct","acct_name")) and "account" in [w.lower() for w in candidates]:
            return c
    raise KeyError(f"Missing required column (tried {list(candidates)}) in columns: {list(df.columns)}")

def _detect_col_qp(df: pd.DataFrame) -> t.Optional[str]:
    for base in ("qp", "quest", "quest_points"):
        try:
            return _detect_col(df, [base])
        except KeyError:
            pass
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl == "qp":
            return c
        if "quest" in cl and ("point" in cl or "total" in cl or "qp" in cl):
            return c
        if "qp" in cl:
            return c
    return None

# ======================================================================================
# Leaderboard normalization
# ======================================================================================

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

def normalize_leaderboard(sheets: t.Dict[str, pd.DataFrame]) -> "_Leader":
    name = next(iter(sheets))
    df = sheets[name].copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_player  = _detect_col(df, ["player","players","subject","name"])
    col_account = _detect_col(df, ["account","username","user","owner","handle"])
    col_sp      = _detect_col(df, ["sp","score","points"])
    col_qp      = _detect_col_qp(df)
    col_rank = None
    for cand in ("rank","position"):
        try:
            col_rank = _detect_col(df, [cand]); break
        except KeyError:
            continue

    df[col_player]  = df[col_player].map(_norm_name)
    df[col_account] = df[col_account].map(_strip_user_suffix).map(_norm_name)
    df[col_sp]      = pd.to_numeric(df[col_sp], errors="coerce").fillna(0).astype(int)
    if col_qp:
        df[col_qp]   = pd.to_numeric(df[col_qp], errors="coerce").fillna(0).astype(int)
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

    return t.cast(_Leader, {
        "by_player": dict(by_player),
        "sp_map": dict(sp_map),
        "display_names": display_player,
        "account_display": account_display,
    })

# ======================================================================================
# Holdings & player tags
# ======================================================================================

def _frame_to_holdings(df: pd.DataFrame) -> t.Dict[str, int]:
    df.columns = [str(c).strip() for c in df.columns]
    col_player = _detect_col(df, ["player","players","subject","name"])
    col_sp = None
    for c in ("sp","score","points","count","qty","quantity"):
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
    return {_norm_name(r[col_player]): int(r[col_sp]) for _, r in grp.iterrows()}

def holdings_from_urls(e31: t.Optional[str], dc: t.Optional[str], fe: t.Optional[str],
                       prefer_env_defaults: bool, ud: t.Optional[str]) -> t.Dict[str, t.Dict[str, int]]:
    urls = {
        "Easystreet31": _pick_url(e31, "holdings_e31", prefer_env_defaults),
        "DusterCrusher": _pick_url(dc, "holdings_dc",  prefer_env_defaults),
        "FinkleIsEinhorn": _pick_url(fe, "holdings_fe",  prefer_env_defaults),
        "UpperDuck": _pick_url(ud, "holdings_ud",  prefer_env_defaults),
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

# ======================================================================================
# Rank/buffer helpers
# ======================================================================================

def _lb_family_sp_for(leader: "_Leader", player: str, account: str) -> int:
    p = _norm_key(player)
    a = _canon_key(account)
    return int(leader["sp_map"].get(p, {}).get(a, 0))

def split_multi_subject_players(players_field: str) -> t.List[str]:
    parts = [p.strip() for p in str(players_field or "").split("/") if str(p).strip()]
    return parts if parts else []

def _rank_and_buffer_full_leader(player: str, leader: "_Leader", family_eff: t.Dict[str, int]
) -> t.Tuple[t.Optional[int], t.Optional[int], t.Optional[str], t.Optional[int]]:
    p = _norm_key(player)
    rows = leader["by_player"].get(p, [])
    if not rows and all(v <= 0 for v in family_eff.values()):
        return None, None, None, None

    combined: t.Dict[str, int] = {}
    for r in rows:
        combined[str(r["account"])] = max(int(combined.get(str(r["account"]), 0)), int(r["sp"] or 0))
    for a_disp, sp in family_eff.items():
        combined[_canon_key(a_disp)] = int(max(0, sp))

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

# ======================================================================================
# Family QP (contest scoring 5-3-1)
# ======================================================================================

def _points_for_rank(r: t.Optional[int]) -> int:
    return int(QP_MAP.get(int(r), 0)) if r is not None else 0

def compute_family_qp_derived_531(
    leader: "_Leader",
    family_sp: t.Dict[str, t.Dict[str, int]],
) -> t.Tuple[int, t.Dict[str, int]]:
    """
    Compute family QP under contest rules (5-3-1) across ALL players, given a
    family SP state (per-account per-player). Returns total and per-account breakdown.
    """
    total = 0
    per_acct = {a: 0 for a in FAMILY_ACCOUNTS}

    # union of leaderboard players and any holdings
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

# ======================================================================================
# Leaderboard Delta — JSON + Export
# ======================================================================================

def _delta_rows(today: "_Leader", yday: "_Leader",
                rivals_canon: t.Set[str], min_sp_delta: int) -> t.List[t.Dict[str, t.Any]]:
    def rival_sum(leader: _Leader, p: str) -> int:
        sm = leader["sp_map"].get(p, {})
        return int(sum(int(sm.get(rv, 0)) for rv in rivals_canon))

    players = set(today["sp_map"].keys()) | set(yday["sp_map"].keys())
    rows: t.List[t.Dict[str, t.Any]] = []
    for p in players:
        fam_eff_before = {a: int(yday["sp_map"].get(p, {}).get(_canon_key(a), 0)) for a in FAMILY_ACCOUNTS}
        fam_eff_after  = {a: int(today["sp_map"].get(p, {}).get(_canon_key(a), 0)) for a in FAMILY_ACCOUNTS}
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
            "family_after":  {"account": a2, "sp": sp2, "rank": r2, "buffer": b2},
            "delta_sp": d_sp,
            "delta_rank": d_rank,
            "delta_buffer": d_buf,
            "rivals_sp_before": rv1,
            "rivals_sp_after": rv2,
            "delta_rivals_sp": d_rival
        })
    rows.sort(key=lambda r: (abs(int(r.get("delta_sp") or 0)), abs(int(r.get("delta_buffer") or 0))), reverse=True)
    return rows

@app.post("/leaderboard_delta_by_urls")
def leaderboard_delta_by_urls(req: LeaderboardDeltaReq):
    try:
        today_url = req.leaderboard_today_url or _pick_url(None, "leaderboard", req.prefer_env_defaults)
        yday_url  = req.leaderboard_yesterday_url or _pick_url(None, "leaderboard_yday", req.prefer_env_defaults)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"missing_urls: {e}")

    try:
        today = normalize_leaderboard(fetch_xlsx(today_url))
        yday  = normalize_leaderboard(fetch_xlsx(yday_url))
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"download_failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"parse_failed: {e}")

    rivals_canon: t.Set[str] = set(_canon_key(_strip_user_suffix(r)) for r in (req.rivals or list(SYNDICATE)))
    rows = _delta_rows(today, yday, rivals_canon, req.min_sp_delta)

    MAX_JSON_ROWS = int(os.getenv("DELTA_JSON_ROW_CAP", "1000"))
    return {
        "ok": True,
        "leaderboard_today_url": today_url,
        "leaderboard_yesterday_url": yday_url,
        "params": {"rivals": sorted(list(rivals_canon)), "min_sp_delta": int(req.min_sp_delta)},
        "summary": {"players_scanned": len(set(today['sp_map'].keys()) | set(yday['sp_map'].keys())),
                    "players_reported": min(len(rows), MAX_JSON_ROWS)},
        "players": rows[:MAX_JSON_ROWS]
    }

def _per_account_movement_lists(player_key: str, today: _Leader, yday: _Leader) -> t.Tuple[str, str]:
    t_map = today["sp_map"].get(player_key, {})
    y_map = yday["sp_map"].get(player_key, {})
    acc_disp = dict(today["account_display"])
    for k, v in yday["account_display"].items():
        acc_disp.setdefault(k, v)

    all_acc = set(t_map.keys()) | set(y_map.keys())
    gains: t.List[t.Tuple[int, str]] = []
    losses: t.List[t.Tuple[int, str]] = []

    for a in all_acc:
        before = int(y_map.get(a, 0))
        after = int(t_map.get(a, 0))
        d = after - before
        if d == 0:
            continue
        name = acc_disp.get(a) or _FAMILY_CANON_TO_DISPLAY.get(a) or a
        if d > 0:
            gains.append((d, f"{name}(+{d})"))
        else:
            losses.append((-d, f"{name}(-{abs(d)})"))

    gains.sort(key=lambda x: (-x[0], x[1].lower()))
    losses.sort(key=lambda x: (-x[0], x[1].lower()))
    return "; ".join(g for _, g in gains), "; ".join(l for _, l in losses)

@app.get("/leaderboard_delta_export")
def leaderboard_delta_export(
    prefer_env_defaults: bool = Query(True),
    leaderboard_today_url: t.Optional[str] = None,
    leaderboard_yesterday_url: t.Optional[str] = None,
    rivals: t.Optional[str] = None,
    min_sp_delta: int = Query(1, ge=0),
    format: t.Literal["csv","xlsx"] = Query("csv")
):
    try:
        today_url = leaderboard_today_url or _pick_url(None, "leaderboard", prefer_env_defaults)
        yday_url  = leaderboard_yesterday_url or _pick_url(None, "leaderboard_yday", prefer_env_defaults)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"missing_urls: {e}")

    try:
        today = normalize_leaderboard(fetch_xlsx(today_url))
        yday  = normalize_leaderboard(fetch_xlsx(yday_url))
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"download_failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"parse_failed: {e}")

    rival_list = [r.strip() for r in (rivals.split(",") if rivals else list(SYNDICATE)) if r.strip()]
    rivals_canon: t.Set[str] = set(_canon_key(_strip_user_suffix(r)) for r in rival_list)
    rows = _delta_rows(today, yday, rivals_canon, min_sp_delta)

    flat = []
    for r in rows:
        fb = r["family_before"]; fa = r["family_after"]
        pkey = r["player_key"]
        acc_gain_str, acc_loss_str = _per_account_movement_lists(pkey, today, yday)

        flat.append({
            "player": r["player"],
            "family_before_account": fb["account"],
            "family_before_sp": fb["sp"],
            "family_before_rank": fb["rank"],
            "family_before_buffer": fb["buffer"],
            "family_after_account": fa["account"],
            "family_after_sp": fa["sp"],
            "family_after_rank": fa["rank"],
            "family_after_buffer": fa["buffer"],
            "delta_sp": r["delta_sp"],
            "delta_rank": r["delta_rank"],
            "delta_buffer": r["delta_buffer"],
            "rivals_sp_before": r["rivals_sp_before"],
            "rivals_sp_after": r["rivals_sp_after"],
            "delta_rivals_sp": r["delta_rivals_sp"],
            "accounts_gained": acc_gain_str,
            "accounts_lost": acc_loss_str,
        })

    df = pd.DataFrame(flat)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"leaderboard_delta_{ts}.{format}"

    if format == "csv":
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
        return Response(content=csv_bytes, media_type="text/csv; charset=utf-8", headers=headers)

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="delta")
    bio.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return StreamingResponse(bio, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)

# ======================================================================================
# Evaluate Trade — restored route with 5-3-1 QP deltas
# ======================================================================================

def _route_order_for(tag: t.Optional[str]) -> t.List[str]:
    if tag == "LEGENDS":
        return ["FinkleIsEinhorn", "DusterCrusher", "Easystreet31", "UpperDuck"]
    if tag == "ANA":
        return ["UpperDuck", "Easystreet31", "DusterCrusher", "FinkleIsEinhorn"]
    if tag in ("DAL", "LAK", "PIT"):
        return ["DusterCrusher", "Easystreet31", "FinkleIsEinhorn", "UpperDuck"]
    return ["Easystreet31", "DusterCrusher", "FinkleIsEinhorn", "UpperDuck"]

def _detect_tag(player_key: str, tags: t.Dict[str, t.Set[str]]) -> t.Optional[str]:
    for k in ("LEGENDS","ANA","DAL","LAK","PIT"):
        if player_key in tags.get(k, set()):
            return k
    return None

def _wingnut_guard_meta(leader: _Leader, player_key: str) -> t.Dict[str, t.Any]:
    rows = leader["sp_map"].get(player_key, {})
    if not rows:
        return {"wingnut_guard_applied": False, "allowed": True}
    sorted_rows = sorted([(acc, sp) for acc, sp in rows.items()], key=lambda x: -int(x[1]))
    rank = None
    for i, (acc, _sp) in enumerate(sorted_rows, start=1):
        if acc == _canon_key("Wingnut84"):
            rank = i
            break
    if rank is None:
        return {"wingnut_guard_applied": False, "allowed": True}
    return {"wingnut_guard_applied": True, "allowed": True, "wingnut_rank": rank}

@app.post("/family_evaluate_trade_by_urls")
def family_evaluate_trade_by_urls(req: FamilyEvaluateTradeReq):
    # Resolve URLs
    try:
        lb_url = _pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)
        tags_url = _pick_url(req.player_tags_url, "player_tags", req.prefer_env_defaults)
        holds = holdings_from_urls(req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url,
                                   req.prefer_env_defaults, req.holdings_ud_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"missing_urls: {e}")

    # Load leaderboard + tags
    try:
        leader = normalize_leaderboard(fetch_xlsx(lb_url))
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"download_failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"parse_failed: {e}")

    try:
        tags = _load_player_tags(req.prefer_env_defaults, tags_url)
    except Exception:
        tags = {"LEGENDS": set(), "ANA": set(), "DAL": set(), "LAK": set(), "PIT": set()}

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

    # Accumulate intended GET and GIVE by player
    get_add: t.Dict[str, int] = defaultdict(int)
    give_subtract: t.Dict[str, int] = defaultdict(int)
    for line in req.trade:
        subjects = split_multi_subject_players(line.players) if req.multi_subject_rule == "full_each_unique" else [line.players]
        for subj in subjects:
            pkey = _norm_key(subj)
            if line.side == "GET":
                get_add[pkey] += int(line.sp)
            else:
                give_subtract[pkey] += int(line.sp)

    # Decide routing for GETs
    allocation_plan: t.List[t.Dict[str, t.Any]] = []
    routed_to: t.Dict[str, t.Tuple[str,int]] = {}  # player_key -> (account, add_sp)
    for pkey, add_sp in get_add.items():
        tag = _detect_tag(pkey, tags)
        order = _route_order_for(tag)
        to = None
        if tag is None:
            best_sp = -1
            for a in FAMILY_ACCOUNTS:
                cur = int(fam_before[a].get(pkey, 0))
                if cur > best_sp or (cur == best_sp and to is not None and order.index(a) < order.index(to)):
                    best_sp = cur
                    to = a
            if to is None:
                to = order[0]
        else:
            to = order[0]

        routed_to[pkey] = (to, add_sp)
        trace = {
            "policy": "tag_first_or_best_holder_with_wingnut_guard",
            "order": order,
            "wingnut_guard": _wingnut_guard_meta(leader, pkey),
        }
        allocation_plan.append({"type":"GET","to":to,"players":[leader["display_names"].get(pkey, pkey)],"sp":int(add_sp),"routing_trace":trace})

    # Apply deltas to produce fam_after
    fam_after: t.Dict[str, t.Dict[str, int]] = {a: dict(spmap) for a, spmap in fam_before.items()}
    # GET adds
    for pkey, (to, add_sp) in routed_to.items():
        fam_after[to][pkey] = int(fam_after[to].get(pkey, 0)) + int(add_sp)
    # GIVE subtracts (always on trade_account)
    for pkey, sub_sp in give_subtract.items():
        fam_after[req.trade_account][pkey] = int(fam_after[req.trade_account].get(pkey, 0)) - int(sub_sp)
        if fam_after[req.trade_account][pkey] < 0:
            fam_after[req.trade_account][pkey] = 0  # clamp

    # Compute player changes for touched players, including 5-3-1 QP
    touched = set(get_add.keys()) | set(give_subtract.keys())
    player_changes: t.List[t.Dict[str, t.Any]] = []

    for pkey in sorted(touched):
        disp = leader["display_names"].get(pkey, pkey)
        per_before = {a: int(fam_before[a].get(pkey, 0)) for a in FAMILY_ACCOUNTS}
        per_after  = {a: int(fam_after[a].get(pkey, 0)) for a in FAMILY_ACCOUNTS}
        sp_before = sum(per_before.values())
        sp_after  = sum(per_after.values())
        delta_sp = sp_after - sp_before

        r1, b1, _a1, _sp1 = _rank_and_buffer_full_leader(pkey, leader, per_before)
        r2, b2, _a2, _sp2 = _rank_and_buffer_full_leader(pkey, leader, per_after)

        qp_before = _points_for_rank(r1)
        qp_after  = _points_for_rank(r2)
        d_qp = qp_after - qp_before

        change = {
            "player": disp,
            "sp_before": sp_before,
            "sp_after": sp_after,
            "delta_sp": delta_sp,
            "per_account_sp_before": per_before,
            "per_account_sp_after": per_after,
            "per_account_sp_delta": {a: per_after[a]-per_before[a] for a in FAMILY_ACCOUNTS},
            "best_rank_before": r1, "best_rank_after": r2,
            "best_rank_before_label": str(r1) if r1 is not None else "—",
            "best_rank_after_label": str(r2) if r2 is not None else "—",
            "buffer_before": b1, "buffer_after": b2,
            "delta_buffer": (b2 - b1) if (b1 is not None and b2 is not None) else None,
            "qp_before": qp_before, "qp_after": qp_after, "delta_qp": d_qp,
        }
        player_changes.append(change)

    # Totals & verdict
    total_sp = sum(int(p["delta_sp"] or 0) for p in player_changes)
    total_buf = sum(int(p["delta_buffer"] or 0) for p in player_changes if p["delta_buffer"] is not None)
    total_qp = int(sum(int(p["delta_qp"] or 0) for p in player_changes))

    if total_qp > 0:
        verdict = "APPROVE"
    elif total_qp == 0:
        verdict = "CAUTION"
    else:
        verdict = "DECLINE"

    # Family QP snapshot (contest scoring) — compute across ALL players
    fam_qp_before, _ = compute_family_qp_derived_531(leader, fam_before)
    fam_qp_after,  _ = compute_family_qp_derived_531(leader, fam_after)

    # Sanity: keep “TOTALS QP Δ == Family ΔQP”
    # (Your Copilot expects this equality for the QP SUMMARY.)
    assert int(fam_qp_after - fam_qp_before) == total_qp

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
