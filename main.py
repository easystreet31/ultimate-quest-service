# main.py — Ultimate Quest Service (Easystreet31)
# Endpoints:
#   POST /evaluate_by_urls_easystreet31
#   POST /scan_by_urls_easystreet31
#   POST /scan_rival_by_urls_easystreet31
#   POST /review_collection_by_urls_easystreet31
#   POST /scan_partner_by_urls_easystreet31
from typing import List, Literal, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, AnyUrl
import pandas as pd
import numpy as np
import io, re, zipfile, math

SERVICE_VERSION = "3.3.0-partner-opps"

# ---- Robust URL fetch (requests if present; urllib fallback) ----
try:
    import requests
    def _fetch(url: str, timeout: int = 25):
        r = requests.get(url, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r.content, (r.headers.get("content-type") or "").lower(), r.url
except Exception:
    import urllib.request as _urlreq
    def _fetch(url: str, timeout: int = 25):
        req = _urlreq.Request(url)
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            ctype = (resp.headers.get("Content-Type") or "").lower()
            final_url = resp.geturl()
            return data, ctype, final_url

app = FastAPI(title="Ultimate Quest Service (Easystreet31)", version=SERVICE_VERSION)

# -------------------- Models --------------------
class TradeRow(BaseModel):
    side: Literal["GET", "GIVE"]
    players: str
    sp: float

class EvalByUrls(BaseModel):
    leaderboard_url: AnyUrl
    holdings_url: AnyUrl
    trade: List[TradeRow]
    multi_subject_rule: Literal["full_each_unique", "split_even"] = "full_each_unique"
    defend_buffer: int = 20
    scope: Literal["trade_only", "portfolio_union"] = "trade_only"
    max_return_players: int = 200
    players_whitelist: Optional[List[str]] = None

class ScanByUrls(BaseModel):
    leaderboard_url: AnyUrl
    holdings_url: AnyUrl
    defend_buffer: int = 20
    upgrade_gap: int = 12
    entry_gap: int = 8
    keep_buffer: int = 30
    max_each: int = 25
    players_whitelist: Optional[List[str]] = None
    players_exclude: Optional[List[str]] = None

class RivalScanByUrls(BaseModel):
    leaderboard_url: AnyUrl
    holdings_url: AnyUrl
    rival: str
    defend_buffer: int = 20
    upgrade_gap: int = 12
    entry_gap: int = 8
    keep_buffer: int = 30
    max_each: int = 0            # <=0 => no omissions
    players_whitelist: Optional[List[str]] = None
    players_exclude: Optional[List[str]] = None

class CollectionReviewByUrls(BaseModel):
    leaderboard_url: AnyUrl
    holdings_url: AnyUrl
    collection_url: AnyUrl
    multi_subject_rule: Literal["full_each_unique", "split_even"] = "full_each_unique"
    defend_buffer: int = 20
    upgrade_gap: int = 12
    entry_gap: int = 8
    keep_buffer: int = 30
    max_each: int = 25                  # <=0 => show all
    max_multiples_per_card: int = 3
    scan_top_candidates: int = 60
    players_whitelist: Optional[List[str]] = None
    players_exclude: Optional[List[str]] = None
    baseline_trade: Optional[List[TradeRow]] = None
    focus_rival: Optional[str] = None
    rival_only: bool = False
    rival_score_weight: int = 250

class PartnerOppsByUrls(BaseModel):
    leaderboard_url: AnyUrl
    holdings_url: AnyUrl
    partner: str
    target_rivals: Optional[List[str]] = None
    max_sp_to_gain: int = 25
    protect_qp: bool = True
    protect_buffer: Optional[int] = None
    max_each: int = 50                # <=0 => no omissions
    players_whitelist: Optional[List[str]] = None
    players_exclude: Optional[List[str]] = None

# -------------------- JSON sanitization --------------------
def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        df = obj.replace({np.inf: None, -np.inf: None})
        df = df.where(pd.notnull(df), None)
        return [_json_safe(r) for r in df.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        s = obj.replace({np.inf: None, -np.inf: None})
        s = s.where(pd.notnull(s), None)
        return _json_safe(s.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj);  return f if math.isfinite(f) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Timestamp): return obj.isoformat()
    except Exception:
        pass
    try:
        import pandas as _pd
        if _pd.isna(obj): return None
    except Exception:
        pass
    return obj

# -------------------- Loaders --------------------
def _normalize_gsheets_url(url: str, prefer_xlsx: bool = True) -> str:
    if "docs.google.com/spreadsheets" not in url: return url
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m: return url
    doc_id = m.group(1)
    return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=" + ("xlsx" if prefer_xlsx else "csv")

def _looks_like_xlsx(url: str, content_type: str, head: bytes) -> bool:
    if url.lower().endswith((".xlsx", ".xls")): return True
    if "spreadsheetml" in content_type or "ms-excel" in content_type: return True
    return head[:2] == b"PK"

def _fetch_table(url: str) -> pd.DataFrame:
    url = _normalize_gsheets_url(url, prefer_xlsx=True)
    data, ctype, final = _fetch(url, timeout=25)
    buf = io.BytesIO(data)
    if _looks_like_xlsx(final, ctype, data):
        try:
            xls = pd.ExcelFile(buf, engine="openpyxl")
            frames = []
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                if df.empty or (df.dropna(how="all").shape[0] == 0): continue
                df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", na=False)]
                df["__sheet__"] = sheet
                frames.append(df)
            if not frames:
                raise HTTPException(status_code=400, detail="XLSX appears empty or unreadable.")
            return pd.concat(frames, ignore_index=True)
        except zipfile.BadZipFile:
            buf.seek(0)
    for enc in (None, "utf-8", "utf-8-sig", "latin1"):
        try:
            buf.seek(0)
            df = pd.read_csv(buf, on_bad_lines="skip", encoding=enc) if enc else pd.read_csv(buf, on_bad_lines="skip")
            df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", na=False)]
            return df
        except Exception:
            continue
    raise HTTPException(status_code=400, detail="Unable to parse CSV after multiple encoding attempts.")

# -------------------- Normalizers --------------------
def _short_user(u: str) -> str:
    s = str(u or "").strip()
    s = re.sub(r"\s*\(\s*\d+\s*\)\s*$", "", s)
    return re.sub(r"\s+", " ", s).strip()

def _norm_user(u: str) -> str:
    return _short_user(u).lower()

def _qp_from_rank(rank: int) -> int:
    return 5 if rank == 1 else 3 if rank == 2 else 1 if rank == 3 else 0

def _norm_lb(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().dropna(how="all")
    d = d.loc[:, ~d.columns.astype(str).str.contains("^Unnamed", na=False)]
    lower = {c.lower(): c for c in d.columns}

    player_col = lower.get("player") or lower.get("subject") or lower.get("name") or "__sheet__"
    user_col   = lower.get("user") or lower.get("collector") or lower.get("username")
    sp_col     = lower.get("sp") or lower.get("points") or lower.get("score")

    if player_col not in d.columns:
        obj = [c for c in d.columns if d[c].dtype == object]
        if not obj: raise HTTPException(status_code=400, detail="Leaderboard missing a 'player' column.")
        player_col = obj[0]
    if not user_col:
        obj = [c for c in d.columns if d[c].dtype == object and c != player_col]
        if not obj: raise HTTPException(status_code=400, detail="Leaderboard missing a 'user' column.")
        user_col = obj[0]
    if not sp_col:
        nums = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        if not nums: raise HTTPException(status_code=400, detail="Leaderboard missing SP/points column.")
        sp_col = nums[0]

    out = d[[player_col, user_col, sp_col]].copy()
    out.columns = ["player", "user", "sp"]
    out["player"] = out["player"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out["user"]   = out["user"].astype(str).str.strip()
    out["sp"]     = pd.to_numeric(out["sp"], errors="coerce").fillna(0).astype(int)
    out["user_norm"] = out["user"].map(_norm_user)

    out = out.sort_values(["player", "sp"], ascending=[True, False])
    out["rank"] = out.groupby("player")["sp"].rank(method="first", ascending=False).astype(int)
    return out[out["rank"] <= 5][["player", "user", "user_norm", "sp", "rank"]]

def _norm_holdings(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().dropna(how="all")
    d = d.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", na=False)]
    lower = {c.lower(): c for c in d.columns}

    player_col = lower.get("player") or lower.get("subject") or lower.get("name")
    sp_col     = lower.get("sp") or lower.get("points") or lower.get("subject points")
    rank_col   = lower.get("rank") or lower.get("#") or lower.get("place")
    qp_col     = lower.get("qp")

    if not player_col:
        obj = [c for c in d.columns if d[c].dtype == object]
        if not obj: raise HTTPException(status_code=400, detail="Holdings missing a 'player' column.")
        player_col = obj[0]
    if not sp_col:
        nums = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        if not nums: raise HTTPException(status_code=400, detail="Holdings missing an SP column.")
        sp_col = nums[0]

    out = d[[player_col, sp_col]].copy()
    out.columns = ["player", "sp"]
    out["player"] = out["player"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out["sp"]     = pd.to_numeric(out["sp"], errors="coerce").fillna(0).astype(int)
    out["rank"]   = pd.to_numeric(d.get(rank_col, 99), errors="coerce").fillna(99).astype(int) if rank_col in (d.columns if rank_col else []) else 99
    out["qp"]     = pd.to_numeric(d.get(qp_col, out["rank"].map({1:5,2:3,3:1})), errors="coerce").fillna(0).astype(int) if qp_col in (d.columns if qp_col else []) else out["rank"].map({1:5,2:3,3:1}).fillna(0).astype(int)
    out = out.sort_values(["player", "sp"], ascending=[True, False]).drop_duplicates("player", keep="first")
    return out[["player", "sp", "rank", "qp"]]

# -------------------- Rank logic --------------------
def _rank_with_me(lb: pd.DataFrame, player: str, me: str, my_sp: int):
    me_norm = _norm_user(me)
    sub = lb[lb["player"] == player][["user", "user_norm", "sp"]].copy()
    if sub.empty:
        sub = pd.DataFrame([{"user": me, "user_norm": me_norm, "sp": int(my_sp)}])
    else:
        mask_me = sub["user_norm"].eq(me_norm)
        if mask_me.any(): sub.loc[mask_me, "sp"] = int(my_sp)
        else: sub = pd.concat([sub, pd.DataFrame([{"user": me, "user_norm": me_norm, "sp": int(my_sp)}])], ignore_index=True)
    sub = sub.sort_values("sp", ascending=False).reset_index(drop=True)

    def _as_label(u_norm: str, u_raw: str) -> str:
        return "YOU" if u_norm == me_norm else _short_user(u_raw)

    my_idx = sub.index[sub["user_norm"] == me_norm]
    my_rank = int(my_idx[0] + 1) if len(my_idx) else 99

    first_sp  = int(sub.iloc[0]["sp"]) if len(sub) > 0 else 0
    first_usr = _as_label(sub.iloc[0]["user_norm"], sub.iloc[0]["user"]) if len(sub) > 0 else None
    first_norm= str(sub.iloc[0]["user_norm"]) if len(sub) > 0 else None
    second_sp = int(sub.iloc[1]["sp"]) if len(sub) > 1 else 0
    second_usr= _as_label(sub.iloc[1]["user_norm"], sub.iloc[1]["user"]) if len(sub) > 1 else None
    second_norm= str(sub.iloc[1]["user_norm"]) if len(sub) > 1 else None
    third_sp  = int(sub.iloc[2]["sp"]) if len(sub) > 2 else 0
    third_usr = _as_label(sub.iloc[2]["user_norm"], sub.iloc[2]["user"]) if len(sub) > 2 else None
    third_norm= str(sub.iloc[2]["user_norm"]) if len(sub) > 2 else None

    return (my_rank, first_sp, second_sp, third_sp,
            first_usr, second_usr, third_usr,
            first_norm, second_norm, third_norm)

# -------------------- Trade → deltas --------------------
def _trade_to_deltas(rows: Optional[List[TradeRow]], rule: str = "full_each_unique") -> Dict[str, float]:
    agg: Dict[str, float] = {}
    if not rows: return agg
    for r in rows:
        players_raw = re.sub(r"\s+", " ", r.players).strip()
        names = [p.strip() for p in players_raw.split("/") if p.strip()]
        if rule == "full_each_unique":
            names = list(dict.fromkeys(names)); per_player_sp = float(r.sp)
        else:
            per_player_sp = float(r.sp) / max(1, len(names))
        sign = +1.0 if r.side == "GET" else -1.0
        for n in names:
            agg[n] = agg.get(n, 0.0) + sign * per_player_sp
    return agg

# -------------------- Bundle (N copies of one card) → deltas --------------------
def _bundle_to_deltas(bundle: List[Tuple[List[str], int]], rule: str = "full_each_unique") -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for players, sp in bundle:
        if rule == "full_each_unique":
            names = list(dict.fromkeys(players))
            per = float(sp)
        else:
            names = list(players)
            per = float(sp) / max(1, len(names))
        for n in names:
            deltas[n] = deltas.get(n, 0.0) + per
    return deltas

# -------------------- Evaluate a set of deltas --------------------
def _evaluate(lb_df: pd.DataFrame,
              hold_df: pd.DataFrame,
              deltas: Dict[str, float],
              me: str,
              defend_buffer: int,
              scope: str,
              max_return_players: int,
              players_whitelist: Optional[List[str]]) -> Dict[str, Any]:

    delta_keys = set(deltas.keys())
    players_all = sorted(delta_keys) if scope == "trade_only" else sorted(set(hold_df["player"].tolist()) | delta_keys)
    if players_whitelist:
        wl = {re.sub(r"\s+", " ", p).strip() for p in players_whitelist}
        players_all = [p for p in players_all if p in wl]

    before_rows, after_rows = [], []
    for p in players_all:
        you_sp_before = int(hold_df.loc[hold_df["player"] == p, "sp"].iloc[0]) if (hold_df["player"] == p).any() else 0

        r, f, s, t, fu, su, tu, fn, sn, tn = _rank_with_me(lb_df, p, me, you_sp_before)
        before_rows.append({"player": p, "you_sp": you_sp_before, "rank": r, "qp": _qp_from_rank(r),
                            "first_sp": f, "second_sp": s, "third_sp": t})

        you_sp_after = int(you_sp_before + int(round(deltas.get(p, 0))))
        r2, f2, s2, t2, fu2, su2, tu2, fn2, sn2, tn2 = _rank_with_me(lb_df, p, me, you_sp_after)
        after_rows.append({"player": p, "you_sp_after": you_sp_after, "rank_after": r2, "qp_after": _qp_from_rank(r2),
                           "first_sp_after": f2, "second_sp_after": s2, "third_sp_after": t2})

    before_df = pd.DataFrame(before_rows)
    after_df  = pd.DataFrame(after_rows)
    cmp = pd.merge(before_df, after_df, on="player", how="outer").fillna(0)

    cmp["qp_delta"] = cmp["qp_after"] - cmp["qp"]
    cmp["sp_delta"] = cmp["you_sp_after"] - cmp["you_sp"]

    cmp["margin_before"] = cmp.apply(lambda r: (r["you_sp"] - r["second_sp"]) if r["rank"] == 1 else None, axis=1)
    cmp["margin_after"]  = cmp.apply(lambda r: (r["you_sp_after"] - r["second_sp_after"]) if r["rank_after"] == 1 else None, axis=1)
    cmp["created_thin_lead"] = cmp.apply(
        lambda r: 1 if (r["rank_after"] == 1 and (r["margin_after"] is not None) and (r["margin_after"] <= defend_buffer)
                        and (r["margin_before"] is None or r["margin_before"] > defend_buffer)) else 0, axis=1)
    cmp["lost_first_place"] = cmp.apply(lambda r: 1 if (r["rank"] == 1 and r["rank_after"] != 1) else 0, axis=1)

    qp_total_before = int(before_df["qp"].sum())
    qp_total_after  = int(after_df["qp_after"].sum())
    qp_delta_total  = qp_total_after - qp_total_before

    verdict = "GREEN" if (qp_delta_total > 0 and cmp["lost_first_place"].sum() == 0 and cmp["created_thin_lead"].sum() == 0) \
              else ("AMBER" if qp_delta_total >= 0 else "RED")

    touched = cmp[(cmp["sp_delta"] != 0) | (cmp["qp_delta"] != 0) |
                  (cmp["created_thin_lead"] == 1) | (cmp["lost_first_place"] == 1) |
                  (cmp["player"].isin(players_all))].copy()

    touched = touched.sort_values(["qp_delta", "sp_delta"], ascending=[False, False])
    per_player = touched[["player","you_sp","you_sp_after","sp_delta","rank","rank_after","qp","qp_after","qp_delta",
                          "margin_before","margin_after","created_thin_lead","lost_first_place"]]

    omitted = 0
    if len(per_player) > max_return_players:
        omitted = len(per_player) - max_return_players
        per_player = per_player.head(max_return_players)

    return {
        "portfolio_qp_before": qp_total_before,
        "portfolio_qp_after":  qp_total_after,
        "portfolio_qp_delta":  qp_delta_total,
        "buffer_target":       defend_buffer,
        "risks": { "lost_firsts": int(cmp["lost_first_place"].sum()),
                   "created_thin_leads": int(cmp["created_thin_lead"].sum()) },
        "per_player": per_player,
        "omitted_players": omitted,
        "verdict": verdict
    }

# -------------------- Daily scans --------------------
def _scan(lb_df: pd.DataFrame,
          hold_df: pd.DataFrame,
          me: str,
          defend_buffer: int,
          upgrade_gap: int,
          entry_gap: int,
          keep_buffer: int,
          max_each: int,
          players_whitelist: Optional[List[str]],
          players_exclude: Optional[List[str]]) -> Dict[str, Any]:

    players_all = sorted(set(hold_df["player"].tolist()) | set(lb_df["player"].tolist()))
    norm = lambda s: re.sub(r"\s+", " ", s).strip()

    if players_whitelist:
        wl = {norm(p) for p in players_whitelist};  players_all = [p for p in players_all if norm(p) in wl]
    if players_exclude:
        ex = {norm(p) for p in players_exclude};    players_all = [p for p in players_all if norm(p) not in ex]

    rows = []
    for p in players_all:
        you_sp = int(hold_df.loc[hold_df["player"] == p, "sp"].iloc[0]) if (hold_df["player"] == p).any() else 0
        r, f, s, t, fu, su, tu, fn, sn, tn = _rank_with_me(lb_df, p, me, you_sp)
        rows.append({"player": p, "you_sp": you_sp, "you_rank": r,
                     "first_sp": f, "second_sp": s, "third_sp": t,
                     "first_user": fu, "second_user": su, "third_user": tu})
    snap = pd.DataFrame(rows)

    thin = snap[snap["you_rank"] == 1].copy()
    thin["margin"] = thin["you_sp"] - thin["second_sp"]
    thin = thin[thin["margin"] <= defend_buffer]
    thin["add_to_buffer"] = (defend_buffer + 1 - thin["margin"]).clip(lower=0).astype(int)
    thin = thin.sort_values("margin", ascending=True)[["player","you_sp","second_user","second_sp","margin","add_to_buffer"]]

    upg = snap[snap["you_rank"] == 2].copy()
    upg["gap_to_1st"] = upg["first_sp"] - upg["you_sp"]
    upg = upg[upg["gap_to_1st"] <= upgrade_gap]
    upg["add_to_take_1st"] = (upg["gap_to_1st"] + 1).clip(lower=0).astype(int)
    upg = upg.sort_values("gap_to_1st", ascending=True)[["player","you_sp","first_user","first_sp","gap_to_1st","add_to_take_1st"]]

    entry = snap[snap["you_rank"] > 3].copy()
    entry["gap_to_3rd"] = entry["third_sp"] - entry["you_sp"]
    entry = entry[(entry["gap_to_3rd"] >= 0) & (entry["gap_to_3rd"] <= entry_gap)]
    entry["add_to_enter_top3"] = (entry["gap_to_3rd"] + 1).astype(int)
    entry = entry.sort_values("gap_to_3rd", ascending=True)[["player","you_sp","third_user","third_sp","gap_to_3rd","add_to_enter_top3"]]

    over = snap[snap["you_rank"] == 1].copy()
    over["margin"] = over["you_sp"] - over["second_sp"]
    overshoot = over[over["margin"] > keep_buffer].copy()
    overshoot["tradable_slack"] = (overshoot["margin"] - keep_buffer).astype(int)
    overshoot = overshoot.sort_values("tradable_slack", ascending=False)[["player","you_sp","second_user","second_sp","margin","tradable_slack"]]

    rivals = pd.concat([
        thin["second_user"].dropna().astype(str).str.strip(),
        upg["first_user"].dropna().astype(str).str.strip(),
        entry["third_user"].dropna().astype(str).str.strip()
    ], ignore_index=True)
    rivals = rivals[rivals.str.upper() != "YOU"]
    rival_counts = rivals.value_counts().reset_index()
    rival_counts.columns = ["rival_user", "mentions"]

    def cap_df(df):
        omitted = max(0, len(df) - max_each)
        return df.head(max_each), omitted

    thin_c, thin_om = cap_df(thin)
    upg_c,  upg_om  = cap_df(upg)
    ent_c,  ent_om  = cap_df(entry)
    ovr_c,  ovr_om  = cap_df(overshoot)
    rv_c,   rv_om   = cap_df(rival_counts)

    return {
        "params": { "defend_buffer": defend_buffer, "upgrade_gap": upgrade_gap,
                    "entry_gap": entry_gap, "keep_buffer": keep_buffer, "max_each": max_each },
        "counts": { "thin": len(thin), "upgrades": len(upg),
                    "top3_entries": len(entry), "overshoots": len(overshoot), "rivals": len(rival_counts) },
        "thin_leads": thin_c, "omitted_thin": thin_om,
        "upgrade_opps": upg_c, "omitted_upgrades": upg_om,
        "top3_entries": ent_c, "omitted_top3": ent_om,
        "overshoots": ovr_c, "omitted_overshoots": ovr_om,
        "rival_watchlist": rv_c, "omitted_rivals": rv_om
    }

def _scan_vs_rival(lb_df: pd.DataFrame,
                   hold_df: pd.DataFrame,
                   me: str, rival: str,
                   defend_buffer: int, upgrade_gap: int, entry_gap: int, keep_buffer: int,
                   max_each: int,
                   players_whitelist: Optional[List[str]],
                   players_exclude: Optional[List[str]]) -> Dict[str, Any]:

    rival_norm = _norm_user(rival)
    rival_label = _short_user(rival)

    players_all = sorted(set(hold_df["player"].tolist()) | set(lb_df["player"].tolist()))
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    if players_whitelist:
        wl = {norm(p) for p in players_whitelist}; players_all = [p for p in players_all if norm(p) in wl]
    if players_exclude:
        ex = {norm(p) for p in players_exclude};   players_all = [p for p in players_all if norm(p) not in ex]

    thin_rows, upg_rows, ent_rows, ovr_rows = [], [], [], []
    for p in players_all:
        you_sp = int(hold_df.loc[hold_df["player"] == p, "sp"].iloc[0]) if (hold_df["player"] == p).any() else 0
        r, f, s, t, fu, su, tu, fn, sn, tn = _rank_with_me(lb_df, p, me, you_sp)

        if r == 1 and sn == rival_norm:
            margin = you_sp - s
            addbuf = max(0, defend_buffer + 1 - margin)
            if margin <= defend_buffer:
                thin_rows.append({"player": p, "you_sp": you_sp, "rival": rival_label, "rival_sp": s,
                                  "margin": int(margin), "add_to_buffer": int(addbuf)})
            if margin > keep_buffer:
                ovr_rows.append({"player": p, "you_sp": you_sp, "rival": rival_label, "rival_sp": s,
                                 "margin": int(margin), "tradable_slack": int(margin - keep_buffer)})

        if r == 2 and fn == rival_norm:
            gap = f - you_sp
            if gap <= upgrade_gap:
                upg_rows.append({"player": p, "you_sp": you_sp, "rival": rival_label, "rival_sp": f,
                                 "gap_to_rival": int(gap), "add_to_take_1st": int(gap + 1)})

        if r > 3 and tn == rival_norm:
            gap3 = t - you_sp
            if 0 <= gap3 <= entry_gap:
                ent_rows.append({"player": p, "you_sp": you_sp, "rival": rival_label, "rival_sp": t,
                                 "gap_to_3rd": int(gap3), "add_to_enter_top3": int(gap3 + 1)})

    def cap(lst):
        if max_each is None or max_each <= 0:
            return lst, 0
        omitted = max(0, len(lst) - max_each)
        return lst[:max_each], omitted

    thin_c, thin_om = cap(sorted(thin_rows, key=lambda r: r["margin"]))
    upg_c,  upg_om  = cap(sorted(upg_rows,  key=lambda r: r["gap_to_rival"]))
    ent_c,  ent_om  = cap(sorted(ent_rows,  key=lambda r: r["gap_to_3rd"]))
    ovr_c,  ovr_om  = cap(sorted(ovr_rows,  key=lambda r: r["tradable_slack"], reverse=True))

    return {
        "params": { "rival": rival_label,
                    "defend_buffer": defend_buffer, "upgrade_gap": upgrade_gap,
                    "entry_gap": entry_gap, "keep_buffer": keep_buffer, "max_each": max_each },
        "counts": { "thin_vs_rival": len(thin_rows), "upgrade_vs_rival": len(upg_rows),
                    "entry_vs_rival": len(ent_rows), "overshoot_vs_rival": len(ovr_rows) },
        "thin_vs_rival": thin_c, "omitted_thin": thin_om,
        "upgrade_vs_rival": upg_c, "omitted_upgrades": upg_om,
        "entry_vs_rival": ent_c, "omitted_top3": ent_om,
        "overshoot_vs_rival": ovr_c, "omitted_overshoots": ovr_om
    }

# -------------------- Collection normalizer --------------------
def _norm_collection(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().dropna(how="all")
    d = d.loc[:, ~d.columns.astype(str).str.contains("^Unnamed", na=False)]
    lower = {c.lower(): c for c in d.columns}

    player_col = lower.get("player") or lower.get("players") or lower.get("subject") or lower.get("name")
    sp_col     = lower.get("sp") or lower.get("subject points") or lower.get("points")
    qty_col    = lower.get("their qty") or lower.get("qty") or lower.get("quantity") or lower.get("count")
    name_col   = lower.get("card name") or lower.get("name")
    num_col    = lower.get("card number") or lower.get("card #") or lower.get("no") or lower.get("number")

    if not player_col or not sp_col:
        raise HTTPException(status_code=400, detail="Collection missing required columns (Player, SP).")
    if not qty_col:
        d["__qty__"] = 1; qty_col = "__qty__"
    if not num_col:
        d["__id__"] = d.get(name_col, pd.Series([""] * len(d))); num_col = "__id__"

    out = d[[player_col, sp_col, qty_col, name_col, num_col]].copy()
    out.columns = ["players_raw", "sp", "qty", "card_name", "card_number"]

    out["sp"] = pd.to_numeric(out["sp"], errors="coerce").fillna(0).astype(int)
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").fillna(0).astype(int)
    out["players_raw"] = out["players_raw"].astype(str).str.strip()
    out["players"] = out["players_raw"].apply(lambda s: [p.strip() for p in s.split("/") if p.strip()])
    out["players"] = out["players"].apply(lambda li: list(dict.fromkeys(li)))

    grp = out.groupby(["card_number", "card_name", "sp"], as_index=False)["qty"].sum()
    first_players = out.groupby(["card_number", "card_name", "sp"])["players"].first().reset_index()
    merged = pd.merge(grp, first_players, on=["card_number", "card_name", "sp"], how="left")
    merged["card_id"] = merged["card_number"].astype(str) + " — " + merged["card_name"].astype(str)
    merged = merged[merged["qty"] > 0].reset_index(drop=True)
    return merged[["card_id", "card_number", "card_name", "players", "sp", "qty"]]

# -------------------- Effects summarizer --------------------
def _summarize_effect(eval_result: Dict[str, Any], defend_buffer: int) -> Dict[str, Any]:
    per = pd.DataFrame(eval_result["per_player"])
    promos_first, promos_top3, shored = [], [], []
    if not per.empty:
        for _, r in per.iterrows():
            before, after = int(r["rank"]), int(r["rank_after"])
            mb, ma = r.get("margin_before"), r.get("margin_after")
            if before != 1 and after == 1: promos_first.append(str(r["player"]))
            if before > 3 and after <= 3:  promos_top3.append(str(r["player"]))
            if before == 1 and (mb is not None) and (ma is not None) and (mb <= defend_buffer) and (ma > defend_buffer):
                shored.append(str(r["player"]))
    return {
        "promotions_to_first": sorted(list(dict.fromkeys(promos_first))),
        "entries_top3":        sorted(list(dict.fromkeys(promos_top3))),
        "shored_thin_firsts":  sorted(list(dict.fromkeys(shored))),
        "lost_firsts":         int(eval_result["risks"]["lost_firsts"]),
        "created_thin_leads":  int(eval_result["risks"]["created_thin_leads"]),
        "qp_delta":            int(eval_result["portfolio_qp_delta"])
    }

# -------------------- Rival-aware impact (per player) --------------------
def _rival_impact_for_player(lb_df: pd.DataFrame,
                             me: str,
                             rival_norm: Optional[str],
                             player: str,
                             you_sp_before: int,
                             you_sp_after: int,
                             defend_buffer: int) -> Dict[str, Any]:
    if not rival_norm:
        return {"took_first": False, "shored_vs_rival": False, "entered_top3_over_rival": False}

    r1, f1, s1, t1, fu1, su1, tu1, fn1, sn1, tn1 = _rank_with_me(lb_df, player, me, you_sp_before)
    r2, f2, s2, t2, fu2, su2, tu2, fn2, sn2, tn2 = _rank_with_me(lb_df, player, me, you_sp_after)

    took_first = (r1 == 2 and fn1 == rival_norm and r2 == 1)

    shored = False
    if r1 == 1 and sn1 == rival_norm:
        margin_before = you_sp_before - s1
        if sn2 == rival_norm:
            margin_after = you_sp_after - s2
        elif tn2 == rival_norm:
            margin_after = you_sp_after - t2
        else:
            margin_after = defend_buffer + 1
        shored = (margin_before <= defend_buffer and margin_after > defend_buffer)

    entered_top3_over = (r1 > 3 and tn1 == rival_norm and r2 <= 3)

    return {
        "took_first": bool(took_first),
        "shored_vs_rival": bool(shored),
        "entered_top3_over_rival": bool(entered_top3_over)
    }

# -------------------- Collection review (qty-aware, rival-aware, fast) --------------------
def _review_collection(lb_df: pd.DataFrame,
                       hold_df: pd.DataFrame,
                       coll_df: pd.DataFrame,
                       me: str,
                       defend_buffer: int,
                       upgrade_gap: int,
                       entry_gap: int,
                       keep_buffer: int,
                       max_each: int,
                       max_multiples_per_card: int,
                       scan_top_candidates: int,
                       multi_subject_rule: str,
                       players_whitelist: Optional[List[str]],
                       players_exclude: Optional[List[str]],
                       baseline_trade: Optional[List[TradeRow]],
                       focus_rival: Optional[str],
                       rival_only: bool,
                       rival_score_weight: int) -> Dict[str, Any]:

    baseline_deltas = _trade_to_deltas(baseline_trade, rule=multi_subject_rule) if baseline_trade else {}

    def _intersects_focus(players: List[str]) -> bool:
        if not players_whitelist and not players_exclude: return True
        n = lambda s: re.sub(r"\s+", " ", s).strip()
        if players_whitelist:
            wl = {n(p) for p in players_whitelist}
            if not any(n(x) in wl for x in players): return False
        if players_exclude:
            ex = {n(p) for p in players_exclude}
            if any(n(x) in ex for x in players): return False
        return True

    candidates = coll_df[coll_df["players"].apply(_intersects_focus)].copy()
    total_cards = int(candidates["qty"].sum())
    total_types = int(len(candidates))

    rival_norm = _norm_user(focus_rival) if focus_rival else None

    scored: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    skipped_no_rival: int = 0

    for _, row in candidates.iterrows():
        try:
            players = row["players"]
            sp      = int(row["sp"])
            qty     = int(row["qty"])
            card_id, card_no, card_name = row["card_id"], row["card_number"], row["card_name"]

            if not isinstance(players, list) or len(players) == 0 or qty <= 0 or sp <= 0:
                continue

            best = None
            k_max = max(1, min(qty, int(max_multiples_per_card)))

            for k in range(1, k_max + 1):
                try:
                    bundle = [(players, sp)] * k
                    deltas = _bundle_to_deltas(bundle, rule=multi_subject_rule)
                    for p, v in baseline_deltas.items():
                        deltas[p] = deltas.get(p, 0.0) + v

                    # Evaluate only touched players for speed
                    res = _evaluate(lb_df, hold_df, deltas, me="Easystreet31",
                                    defend_buffer=defend_buffer, scope="trade_only",
                                    max_return_players=60, players_whitelist=list(deltas.keys()))
                    summary = _summarize_effect(res, defend_buffer=defend_buffer)

                    rival_hits = {"took_first_from_rival": [], "shored_vs_rival": [], "entered_top3_over_rival": []}
                    if rival_norm:
                        for p in deltas.keys():
                            you_sp_before = int(hold_df.loc[hold_df["player"] == p, "sp"].iloc[0]) if (hold_df["player"] == p).any() else 0
                            you_sp_after  = int(you_sp_before + int(round(deltas.get(p, 0))))
                            rfx = _rival_impact_for_player(lb_df, me="Easystreet31", rival_norm=rival_norm,
                                                           player=p, you_sp_before=you_sp_before, you_sp_after=you_sp_after,
                                                           defend_buffer=defend_buffer)
                            if rfx["took_first"]: rival_hits["took_first_from_rival"].append(p)
                            if rfx["shored_vs_rival"]: rival_hits["shored_vs_rival"].append(p)
                            if rfx["entered_top3_over_rival"]: rival_hits["entered_top3_over_rival"].append(p)

                    score = (summary["qp_delta"] * 1000
                             + len(summary["promotions_to_first"]) * 50
                             + len(summary["entries_top3"]) * 20
                             + len(summary["shored_thin_firsts"]) * 10
                             - summary["lost_firsts"] * 200
                             - summary["created_thin_leads"] * 30)

                    if rival_norm:
                        score += (rival_score_weight * len(rival_hits["took_first_from_rival"])
                                  + 100 * len(rival_hits["shored_vs_rival"])
                                  + 60  * len(rival_hits["entered_top3_over_rival"]))

                    item = {
                        "card_id": card_id,
                        "card_number": card_no,
                        "card_name": card_name,
                        "players": players,
                        "sp": sp,
                        "available_qty": qty,
                        "take_n": k,
                        "score": int(score),
                        "rival_score": int(score if rival_norm else 0),
                        "effects": summary,
                        "rival_effects": rival_hits
                    }

                    if rival_only:
                        rival_interacts = any(len(v) > 0 for v in rival_hits.values())
                        if not rival_interacts:
                            continue

                    if (best is None) or (item["score"] > best["score"]):
                        best = item

                except Exception as inner_e:
                    errors.append({"card": f"{card_no} — {card_name}", "take_n": str(k), "error": str(inner_e)[:220]})
                    continue

            if rival_only and best is None:
                skipped_no_rival += 1
            if best:
                scored.append(best)

        except Exception as outer_e:
            errors.append({"card": f"{row.get('card_number','?')} — {row.get('card_name','?')}", "take_n": "n/a", "error": str(outer_e)[:220]})
            continue

    scored_sorted = sorted(scored, key=lambda r: (r["effects"]["qp_delta"], r["score"]), reverse=True)
    if scan_top_candidates and scan_top_candidates > 0:
        scored_sorted = scored_sorted[:int(scan_top_candidates)]

    def cap(lst):
        if max_each is None or max_each <= 0:
            return lst, 0
        omitted = max(0, len(lst) - max_each)
        return lst[:max_each], omitted

    qp_positive = [r for r in scored_sorted if r["effects"]["qp_delta"] > 0]
    thin_shore  = [r for r in scored_sorted if r["effects"]["qp_delta"] >= 0 and len(r["effects"]["shored_thin_firsts"]) > 0]
    take_first  = [r for r in scored_sorted if len(r["effects"]["promotions_to_first"]) > 0]
    enter_top3  = [r for r in scored_sorted if len(r["effects"]["entries_top3"]) > 0]
    rival_prio  = [r for r in scored_sorted if focus_rival and any(len(v) > 0 for v in r["rival_effects"].values())]
    rival_prio  = sorted(rival_prio, key=lambda r: (len(r["rival_effects"]["took_first_from_rival"]),
                                                    len(r["rival_effects"]["shored_vs_rival"]),
                                                    len(r["rival_effects"]["entered_top3_over_rival"]),
                                                    r["score"]), reverse=True)

    pos_c,  pos_om  = cap(qp_positive)
    sho_c,  sho_om  = cap(thin_shore)
    fir_c,  fir_om  = cap(take_first)
    top3_c, top3_om = cap(enter_top3)
    riv_c,  riv_om  = cap(rival_prio)

    error_samples = errors[:5]

    return {
        "params": {
            "defend_buffer": defend_buffer, "upgrade_gap": upgrade_gap, "entry_gap": entry_gap,
            "keep_buffer": keep_buffer, "max_each": max_each,
            "max_multiples_per_card": max_multiples_per_card, "scan_top_candidates": scan_top_candidates,
            "focus_rival": focus_rival, "rival_only": rival_only, "rival_score_weight": rival_score_weight
        },
        "collection_counts": {"card_types": total_types, "total_units": total_cards},
        "qp_positive_picks": pos_c, "omitted_qp_positive": pos_om,
        "shore_thin_from_collection": sho_c, "omitted_shore_thin": sho_om,
        "take_first_from_collection": fir_c, "omitted_take_first": fir_om,
        "enter_top3_from_collection": top3_c, "omitted_enter_top3": top3_om,
        "rival_priority_picks": riv_c, "omitted_rival_priority": riv_om,
        "diagnostics": {
            "errors_count": len(errors),
            "error_samples": error_samples,
            "skipped_because_rival_only": skipped_no_rival
        }
    }

# -------------------- Partner QP Finder --------------------
def _scan_partner_opps(lb_df: pd.DataFrame,
                       hold_df: pd.DataFrame,
                       me: str,
                       partner: str,
                       target_rivals: Optional[List[str]],
                       max_sp_to_gain: int,
                       protect_qp: bool,
                       protect_buffer: Optional[int],
                       max_each: int,
                       players_whitelist: Optional[List[str]],
                       players_exclude: Optional[List[str]]) -> Dict[str, Any]:

    me_norm = _norm_user(me)
    partner_norm = _norm_user(partner)
    rivals_norm = { _norm_user(r) for r in (target_rivals or []) }

    players_all = sorted(set(hold_df["player"].tolist()) | set(lb_df["player"].tolist()))
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    if players_whitelist:
        wl = {norm(p) for p in players_whitelist}; players_all = [p for p in players_all if norm(p) in wl]
    if players_exclude:
        ex = {norm(p) for p in players_exclude};   players_all = [p for p in players_all if norm(p) not in ex]

    def qp_for_index(i: int) -> int:
        return 5 if i == 0 else 3 if i == 1 else 1 if i == 2 else 0

    def rank_list(rows):
        return sorted(rows, key=lambda r: (-int(r["sp"]), r["_order"]))

    def get_rank(ranked, u_norm):
        for i, c in enumerate(ranked):
            if c["user_norm"] == u_norm:
                return i + 1
        return 99

    opportunities = []
    for p in players_all:
        you_sp = int(hold_df.loc[hold_df["player"] == p, "sp"].iloc[0]) if (hold_df["player"] == p).any() else 0

        sub = lb_df[lb_df["player"] == p][["user","user_norm","sp"]].copy()
        comps = []
        for _, r in sub.iterrows():
            comps.append({"user": r["user"], "user_norm": r["user_norm"], "sp": int(r["sp"]), "_order": len(comps)})

        # ensure YOU + partner entries
        present_you = any(c["user_norm"] == me_norm for c in comps)
        present_partner = any(c["user_norm"] == partner_norm for c in comps)
        if present_you:
            for c in comps:
                if c["user_norm"] == me_norm: c["sp"] = you_sp
        else:
            comps.append({"user": "YOU", "user_norm": me_norm, "sp": you_sp, "_order": 1000})
        partner_sp = 0
        if present_partner:
            for c in comps:
                if c["user_norm"] == partner_norm: partner_sp = int(c["sp"])
        else:
            comps.append({"user": partner, "user_norm": partner_norm, "sp": 0, "_order": 1001})

        before = rank_list([dict(c) for c in comps])
        you_rank_before = get_rank(before, me_norm);  yqp_before = qp_for_index(you_rank_before - 1)
        partner_rank_before = get_rank(before, partner_norm); pqp_before = qp_for_index(partner_rank_before - 1)

        first_sp  = before[0]["sp"] if len(before) > 0 else 0
        second_sp = before[1]["sp"] if len(before) > 1 else 0
        third_sp  = before[2]["sp"] if len(before) > 2 else 0

        # try for 3rd, 2nd, 1st (in that order)
        for target_rank, thr in ((3, third_sp), (2, second_sp), (1, first_sp)):
            needed = max(0, int(thr) - int(partner_sp) + 1)
            if needed <= 0:
                continue
            if max_sp_to_gain is not None and needed > int(max_sp_to_gain):
                continue

            after_comps = [dict(c) for c in comps]
            for c in after_comps:
                if c["user_norm"] == partner_norm:
                    c["sp"] = int(c["sp"]) + int(needed)

            after = rank_list(after_comps)
            you_rank_after = get_rank(after, me_norm); yqp_after = qp_for_index(you_rank_after - 1)
            if protect_qp and (yqp_after < yqp_before):
                continue
            if protect_buffer and you_rank_after == 1:
                second_after_sp = after[1]["sp"] if len(after) > 1 else 0
                margin_after = int(you_sp) - int(second_after_sp)
                if margin_after < int(protect_buffer):
                    continue

            partner_rank_after = get_rank(after, partner_norm); pqp_after = qp_for_index(partner_rank_after - 1)
            partner_qp_delta = pqp_after - pqp_before
            if partner_qp_delta <= 0:
                continue

            # who loses QP?
            before_qp = {}
            for i, c in enumerate(before[:3]): before_qp[c["user_norm"]] = qp_for_index(i)
            after_qp  = {}
            for i, c in enumerate(after[:3]):  after_qp[c["user_norm"]]  = qp_for_index(i)

            hurts, hurts_target = [], []
            pool = after + before
            for u_norm, q0 in before_qp.items():
                q1 = after_qp.get(u_norm, 0)
                if q1 < q0 and u_norm not in (me_norm, partner_norm):
                    raw = next((c["user"] for c in pool if c["user_norm"] == u_norm), u_norm)
                    delta = q1 - q0  # negative
                    entry = {"user": _short_user(raw), "qp_change": int(delta)}
                    hurts.append(entry)
                    if u_norm in rivals_norm: hurts_target.append(entry)

            score = (1000 if len(hurts_target) > 0 else 0) + partner_qp_delta * 100 - needed
            opportunities.append({
                "player": p,
                "partner_target_rank": int(partner_rank_after),
                "partner_needed_sp": int(needed),
                "partner_qp_delta": int(partner_qp_delta),
                "your_status": {"rank": int(you_rank_before), "rank_after": int(you_rank_after),
                                "qp": int(yqp_before), "qp_after": int(yqp_after)},
                "hurts": hurts,
                "hurts_target_rival": bool(hurts_target),
                "score": int(score)
            })

    opportunities = sorted(opportunities, key=lambda r: (-int(r["score"]), int(r["partner_needed_sp"])))
    omitted = 0
    if max_each and max_each > 0 and len(opportunities) > max_each:
        omitted = len(opportunities) - max_each
        opportunities = opportunities[:max_each]

    return {
        "params": {
            "partner": partner, "target_rivals": list(target_rivals or []),
            "max_sp_to_gain": max_sp_to_gain, "protect_qp": bool(protect_qp),
            "protect_buffer": protect_buffer, "max_each": max_each
        },
        "counts": {"opportunities": len(opportunities), "omitted": omitted},
        "opportunities": opportunities
    }

# -------------------- Routes --------------------
@app.get("/health")
def health(): return {"ok": True}

@app.get("/info")
def info():
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    return {"version": SERVICE_VERSION, "routes": sorted(routes)}

@app.post("/evaluate_by_urls_easystreet31")
def evaluate_by_urls_easystreet31(payload: EvalByUrls):
    try:
        lb_raw = _fetch_table(str(payload.leaderboard_url))
        hd_raw = _fetch_table(str(payload.holdings_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch leaderboard/holdings: {e}")
    try:
        lb = _norm_lb(lb_raw);  hd = _norm_holdings(hd_raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to normalize inputs: {e}")

    deltas = _trade_to_deltas(payload.trade, rule=payload.multi_subject_rule)
    result = _evaluate(lb, hd, deltas, me="Easystreet31",
                       defend_buffer=payload.defend_buffer,
                       scope=payload.scope,
                       max_return_players=payload.max_return_players,
                       players_whitelist=payload.players_whitelist)
    return JSONResponse(content=_json_safe(result))

@app.post("/scan_by_urls_easystreet31")
def scan_by_urls_easystreet31(payload: ScanByUrls):
    try:
        lb_raw = _fetch_table(str(payload.leaderboard_url))
        hd_raw = _fetch_table(str(payload.holdings_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch leaderboard/holdings: {e}")
    try:
        lb = _norm_lb(lb_raw);  hd = _norm_holdings(hd_raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to normalize inputs: {e}")

    result = _scan(lb, hd, me="Easystreet31",
                   defend_buffer=payload.defend_buffer,
                   upgrade_gap=payload.upgrade_gap,
                   entry_gap=payload.entry_gap,
                   keep_buffer=payload.keep_buffer,
                   max_each=payload.max_each,
                   players_whitelist=payload.players_whitelist,
                   players_exclude=payload.players_exclude)
    return JSONResponse(content=_json_safe(result))

@app.post("/scan_rival_by_urls_easystreet31")
def scan_rival_by_urls_easystreet31(payload: RivalScanByUrls):
    try:
        lb_raw = _fetch_table(str(payload.leaderboard_url))
        hd_raw = _fetch_table(str(payload.holdings_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch leaderboard/holdings: {e}")
    try:
        lb = _norm_lb(lb_raw);  hd = _norm_holdings(hd_raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to normalize inputs: {e}")

    result = _scan_vs_rival(lb, hd, me="Easystreet31", rival=payload.rival,
                            defend_buffer=payload.defend_buffer, upgrade_gap=payload.upgrade_gap,
                            entry_gap=payload.entry_gap, keep_buffer=payload.keep_buffer,
                            max_each=payload.max_each,
                            players_whitelist=payload.players_whitelist,
                            players_exclude=payload.players_exclude)
    return JSONResponse(content=_json_safe(result))

@app.post("/review_collection_by_urls_easystreet31")
def review_collection_by_urls_easystreet31(payload: CollectionReviewByUrls):
    try:
        lb_raw = _fetch_table(str(payload.leaderboard_url))
        hd_raw = _fetch_table(str(payload.holdings_url))
        col_raw = _fetch_table(str(payload.collection_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch leaderboard/holdings/collection: {e}")
    try:
        lb = _norm_lb(lb_raw);  hd = _norm_holdings(hd_raw);  col = _norm_collection(col_raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to normalize inputs: {e}")

    try:
        result = _review_collection(lb_df=lb, hold_df=hd, coll_df=col, me="Easystreet31",
                                    defend_buffer=payload.defend_buffer, upgrade_gap=payload.upgrade_gap,
                                    entry_gap=payload.entry_gap, keep_buffer=payload.keep_buffer,
                                    max_each=payload.max_each, max_multiples_per_card=payload.max_multiples_per_card,
                                    scan_top_candidates=payload.scan_top_candidates,
                                    multi_subject_rule=payload.multi_subject_rule,
                                    players_whitelist=payload.players_whitelist,
                                    players_exclude=payload.players_exclude,
                                    baseline_trade=payload.baseline_trade,
                                    focus_rival=payload.focus_rival,
                                    rival_only=payload.rival_only,
                                    rival_score_weight=payload.rival_score_weight)
        return JSONResponse(content=_json_safe(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collection review failed internally: {e}")

@app.post("/scan_partner_by_urls_easystreet31")
def scan_partner_by_urls_easystreet31(payload: PartnerOppsByUrls):
    try:
        lb_raw = _fetch_table(str(payload.leaderboard_url))
        hd_raw = _fetch_table(str(payload.holdings_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch leaderboard/holdings: {e}")
    try:
        lb = _norm_lb(lb_raw);  hd = _norm_holdings(hd_raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to normalize inputs: {e}")

    result = _scan_partner_opps(lb, hd, me="Easystreet31",
                                partner=payload.partner,
                                target_rivals=payload.target_rivals,
                                max_sp_to_gain=payload.max_sp_to_gain,
                                protect_qp=payload.protect_qp,
                                protect_buffer=payload.protect_buffer,
                                max_each=payload.max_each,
                                players_whitelist=payload.players_whitelist,
                                players_exclude=payload.players_exclude)
    return JSONResponse(content=_json_safe(result))
