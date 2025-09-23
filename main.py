# main.py — Ultimate Quest Service (Easystreet31) — FIX: no suffix KeyError on second_sp
# Endpoints:
#   POST /evaluate_by_urls_easystreet31  (no-quantity trades, trade-scope response)
#   POST /scan_by_urls_easystreet31      (Thin Leads / Upgrade Opps / Top-3 / Overshoots)
from typing import List, Literal, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, AnyUrl
import pandas as pd
import io, re, zipfile

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

app = FastAPI(title="Ultimate Quest Service (Easystreet31)")

# -------------------- Models --------------------
class TradeRow(BaseModel):
    side: Literal["GET", "GIVE"]      # GET = you receive this card; GIVE = you send this card
    players: str                      # e.g., "Ryan O'Reilly/Filip Forsberg" (multi-subject)
    sp: float                         # subject points on the card

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

# -------------------- Normalization helpers --------------------
def _normalize_gsheets_url(url: str, prefer_xlsx: bool = True) -> str:
    if "docs.google.com/spreadsheets" not in url:
        return url
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        return url
    doc_id = m.group(1)
    if prefer_xlsx:
        return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=xlsx"
    return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv"

def _looks_like_xlsx(url: str, content_type: str, head: bytes) -> bool:
    if url.lower().endswith((".xlsx", ".xls")):
        return True
    if "spreadsheetml" in content_type or "ms-excel" in content_type:
        return True
    return head[:2] == b"PK"  # XLSX is a ZIP

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
                if df.empty or (df.dropna(how="all").shape[0] == 0):
                    continue
                df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", na=False)]
                df["__sheet__"] = sheet
                frames.append(df)
            if not frames:
                raise HTTPException(status_code=400, detail="XLSX appears empty or unreadable.")
            return pd.concat(frames, ignore_index=True)
        except zipfile.BadZipFile:
            buf.seek(0)  # fall through to CSV

    # CSV with encodings
    for enc in (None, "utf-8", "utf-8-sig", "latin1"):
        try:
            buf.seek(0)
            df = pd.read_csv(buf, on_bad_lines="skip", encoding=enc) if enc else pd.read_csv(buf, on_bad_lines="skip")
            df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", na=False)]
            return df
        except Exception:
            continue
    raise HTTPException(status_code=400, detail="Unable to parse CSV after multiple encoding attempts.")

def _short_user(u: str) -> str:
    s = str(u or "").strip()
    s = re.sub(r"\s*\(\s*\d+\s*\)\s*$", "", s)  # strip trailing "(digits)"
    return re.sub(r"\s+", " ", s).strip()

def _norm_user(u: str) -> str:
    return _short_user(u).lower()

def _qp_from_rank(rank: int) -> int:
    return 5 if rank == 1 else 3 if rank == 2 else 1 if rank == 3 else 0

# -------------------- Table normalization --------------------
def _norm_lb(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.dropna(how="all")
    d = d.loc[:, ~d.columns.astype(str).str.contains("^Unnamed", na=False)]
    lower = {c.lower(): c for c in d.columns}

    player_col = lower.get("player") or lower.get("subject") or lower.get("name") or "__sheet__"
    user_col   = lower.get("user")   or lower.get("collector") or lower.get("username")
    sp_col     = lower.get("sp")     or lower.get("points") or lower.get("score")

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
    d = df.copy()
    d = d.dropna(how="all")
    d = d.loc[:, ~d.columns.astype(str).str.contains("^Unnamed", na=False)]
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

# -------------------- Rank logic (alias-aware) --------------------
def _rank_with_me(lb: pd.DataFrame, player: str, me: str, my_sp: int):
    me_norm = _norm_user(me)
    sub = lb[lb["player"] == player][["user", "user_norm", "sp"]].copy()

    if sub.empty:
        sub = pd.DataFrame([{"user": me, "user_norm": me_norm, "sp": int(my_sp)}])
    else:
        mask_me = sub["user_norm"].eq(me_norm)
        if mask_me.any():
            sub.loc[mask_me, "sp"] = int(my_sp)
        else:
            sub = pd.concat([sub, pd.DataFrame([{"user": me, "user_norm": me_norm, "sp": int(my_sp)}])], ignore_index=True)

    sub = sub.sort_values("sp", ascending=False).reset_index(drop=True)

    # my position
    my_idx = sub.index[sub["user_norm"] == me_norm]
    my_rank = int(my_idx[0] + 1) if len(my_idx) else 99

    def _as_label(u_norm: str, u_raw: str) -> str:
        return "YOU" if u_norm == me_norm else _short_user(u_raw)

    first_sp  = int(sub.iloc[0]["sp"]) if len(sub) > 0 else 0
    first_usr = _as_label(sub.iloc[0]["user_norm"], sub.iloc[0]["user"]) if len(sub) > 0 else None
    second_sp = int(sub.iloc[1]["sp"]) if len(sub) > 1 else 0
    second_usr= _as_label(sub.iloc[1]["user_norm"], sub.iloc[1]["user"]) if len(sub) > 1 else None
    third_sp  = int(sub.iloc[2]["sp"]) if len(sub) > 2 else 0
    third_usr = _as_label(sub.iloc[2]["user_norm"], sub.iloc[2]["user"]) if len(sub) > 2 else None

    return my_rank, first_sp, second_sp, third_sp, first_usr, second_usr, third_usr

# -------------------- Trade → deltas --------------------
def _trade_to_deltas(rows: List[TradeRow], rule: str = "full_each_unique") -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for r in rows:
        players_raw = re.sub(r"\s+", " ", r.players).strip()
        names = [p.strip() for p in players_raw.split("/") if p.strip()]
        if rule == "full_each_unique":
            names = list(dict.fromkeys(names))
            per_player_sp = float(r.sp)
        else:
            per_player_sp = float(r.sp) / max(1, len(names))
        sign = +1.0 if r.side == "GET" else -1.0
        for n in names:
            agg[n] = agg.get(n, 0.0) + sign * per_player_sp
    return agg

# -------------------- Evaluate trade (FIX: disambiguate second_sp columns) --------------------
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

        # BEFORE
        r, f, s, t, fu, su, tu = _rank_with_me(lb_df, p, me, you_sp_before)
        before_rows.append({
            "player": p, "you_sp": you_sp_before, "rank": r, "qp": _qp_from_rank(r),
            "first_sp": f, "second_sp": s, "third_sp": t
        })

        # AFTER
        you_sp_after = int(you_sp_before + int(round(deltas.get(p, 0))))
        r2, f2, s2, t2, fu2, su2, tu2 = _rank_with_me(lb_df, p, me, you_sp_after)
        after_rows.append({
            "player": p, "you_sp_after": you_sp_after, "rank_after": r2, "qp_after": _qp_from_rank(r2),
            # rename these to avoid merge collisions
            "first_sp_after": f2, "second_sp_after": s2, "third_sp_after": t2
        })

    before_df = pd.DataFrame(before_rows)
    after_df  = pd.DataFrame(after_rows)

    # Merge with distinct column names (no _x/_y suffixes)
    cmp = pd.merge(before_df, after_df, on="player", how="outer").fillna(0)

    # Deltas
    cmp["qp_delta"] = cmp["qp_after"] - cmp["qp"]
    cmp["sp_delta"] = cmp["you_sp_after"] - cmp["you_sp"]

    # Risks (use the correct "second" columns)
    cmp["margin_before"] = cmp.apply(
        lambda r: (r["you_sp"] - r["second_sp"]) if r["rank"] == 1 else None, axis=1
    )
    cmp["margin_after"]  = cmp.apply(
        lambda r: (r["you_sp_after"] - r["second_sp_after"]) if r["rank_after"] == 1 else None, axis=1
    )
    cmp["created_thin_lead"] = cmp.apply(
        lambda r: 1 if (r["rank_after"] == 1 and (r["margin_after"] is not None) and (r["margin_after"] <= defend_buffer)
                        and (r["margin_before"] is None or r["margin_before"] > defend_buffer)) else 0, axis=1)
    cmp["lost_first_place"] = cmp.apply(lambda r: 1 if (r["rank"] == 1 and r["rank_after"] != 1) else 0, axis=1)

    # Trade-scope QP totals
    qp_total_before = int(before_df["qp"].sum())
    qp_total_after  = int(after_df["qp_after"].sum())
    qp_delta_total  = qp_total_after - qp_total_before

    verdict = "GREEN" if (qp_delta_total > 0 and cmp["lost_first_place"].sum() == 0 and cmp["created_thin_lead"].sum() == 0) \
              else ("AMBER" if qp_delta_total >= 0 else "RED")

    # Keep response compact
    touched = cmp[(cmp["sp_delta"] != 0) | (cmp["qp_delta"] != 0) |
                  (cmp["created_thin_lead"] == 1) | (cmp["lost_first_place"] == 1) |
                  (cmp["player"].isin(delta_keys))].copy()

    touched = touched.sort_values(["qp_delta", "sp_delta"], ascending=[False, False])
    per_player = touched[[
        "player","you_sp","you_sp_after","sp_delta","rank","rank_after","qp","qp_after","qp_delta",
        "margin_before","margin_after","created_thin_lead","lost_first_place"
    ]]

    omitted = 0
    if len(per_player) > max_return_players:
        omitted = len(per_player) - max_return_players
        per_player = per_player.head(max_return_players)

    return {
        "portfolio_qp_before": qp_total_before,
        "portfolio_qp_after":  qp_total_after,
        "portfolio_qp_delta":  qp_delta_total,
        "buffer_target":       defend_buffer,
        "risks": {
            "lost_firsts": int(cmp["lost_first_place"].sum()),
            "created_thin_leads": int(cmp["created_thin_lead"].sum()),
        },
        "per_player": per_player.to_dict(orient="records"),
        "omitted_players": omitted,
        "verdict": verdict
    }

# -------------------- Daily scan (alias-aware, compact-friendly) --------------------
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
        wl = {norm(p) for p in players_whitelist}
        players_all = [p for p in players_all if norm(p) in wl]
    if players_exclude:
        ex = {norm(p) for p in players_exclude}
        players_all = [p for p in players_all if norm(p) not in ex]

    rows = []
    for p in players_all:
        you_sp = int(hold_df.loc[hold_df["player"] == p, "sp"].iloc[0]) if (hold_df["player"] == p).any() else 0
        r, f, s, t, fu, su, tu = _rank_with_me(lb_df, p, me, you_sp)
        rows.append({
            "player": p, "you_sp": you_sp, "you_rank": r,
            "first_sp": f, "second_sp": s, "third_sp": t,
            "first_user": fu, "second_user": su, "third_user": tu
        })
    snap = pd.DataFrame(rows)

    # Thin 1sts
    thin = snap[snap["you_rank"] == 1].copy()
    thin["margin"] = thin["you_sp"] - thin["second_sp"]
    thin = thin[thin["margin"] <= defend_buffer]
    thin["add_to_buffer"] = (defend_buffer + 1 - thin["margin"]).clip(lower=0).astype(int)
    thin = thin.sort_values("margin", ascending=True)[
        ["player","you_sp","second_user","second_sp","margin","add_to_buffer"]
    ]

    # Upgrade opps (2nd -> 1st)
    upg = snap[snap["you_rank"] == 2].copy()
    upg["gap_to_1st"] = upg["first_sp"] - upg["you_sp"]
    upg = upg[upg["gap_to_1st"] <= upgrade_gap]
    upg["add_to_take_1st"] = (upg["gap_to_1st"] + 1).clip(lower=0).astype(int)
    upg = upg.sort_values("gap_to_1st", ascending=True)[
        ["player","you_sp","first_user","first_sp","gap_to_1st","add_to_take_1st"]
    ]

    # Top-3 entries
    entry = snap[snap["you_rank"] > 3].copy()
    entry["gap_to_3rd"] = entry["third_sp"] - entry["you_sp"]
    entry = entry[(entry["gap_to_3rd"] >= 0) & (entry["gap_to_3rd"] <= entry_gap)]
    entry["add_to_enter_top3"] = (entry["gap_to_3rd"] + 1).astype(int)
    entry = entry.sort_values("gap_to_3rd", ascending=True)[
        ["player","you_sp","third_user","third_sp","gap_to_3rd","add_to_enter_top3"]
    ]

    # Overshoots
    over = snap[snap["you_rank"] == 1].copy()
    over["margin"] = over["you_sp"] - over["second_sp"]
    overshoot = over[over["margin"] > keep_buffer].copy()
    overshoot["tradable_slack"] = (overshoot["margin"] - keep_buffer).astype(int)
    overshoot = overshoot.sort_values("tradable_slack", ascending=False)[
        ["player","you_sp","second_user","second_sp","margin","tradable_slack"]
    ]

    # Rivals (exclude YOU)
    rivals = pd.concat([
        thin["second_user"].dropna().astype(str).str.strip(),
        upg["first_user"].dropna().astype(str).str.strip(),
        entry["third_user"].dropna().astype(str).str.strip()
    ], ignore_index=True)
    rivals = rivals[rivals.str.upper() != "YOU"]
    rival_counts = rivals.value_counts().reset_index()
    rival_counts.columns = ["rival_user", "mentions"]

    # Cap lists
    def cap(df):
        omitted = max(0, len(df) - max_each)
        return df.head(max_each), omitted

    thin_c, thin_om = cap(thin)
    upg_c, upg_om   = cap(upg)
    ent_c, ent_om   = cap(entry)
    ovr_c, ovr_om   = cap(overshoot)
    rv_c, rv_om     = cap(rival_counts)

    return {
        "params": {
            "defend_buffer": defend_buffer, "upgrade_gap": upgrade_gap,
            "entry_gap": entry_gap, "keep_buffer": keep_buffer, "max_each": max_each
        },
        "counts": {
            "thin": len(thin), "upgrades": len(upg),
            "top3_entries": len(entry), "overshoots": len(overshoot),
            "rivals": len(rival_counts)
        },
        "thin_leads": thin_c.to_dict(orient="records"),
        "omitted_thin": thin_om,
        "upgrade_opps": upg_c.to_dict(orient="records"),
        "omitted_upgrades": upg_om,
        "top3_entries": ent_c.to_dict(orient="records"),
        "omitted_top3": ent_om,
        "overshoots": ovr_c.to_dict(orient="records"),
        "omitted_overshoots": ovr_om,
        "rival_watchlist": rv_c.to_dict(orient="records"),
        "omitted_rivals": rv_om
    }

# -------------------- Routes --------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/evaluate_by_urls_easystreet31")
def evaluate_by_urls_easystreet31(payload: EvalByUrls):
    try:
        lb_raw = _fetch_table(str(payload.leaderboard_url))
        hd_raw = _fetch_table(str(payload.holdings_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch leaderboard/holdings: {e}")
    try:
        lb = _norm_lb(lb_raw)
        hd = _norm_holdings(hd_raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to normalize inputs: {e}")

    deltas = _trade_to_deltas(payload.trade, rule=payload.multi_subject_rule)
    result = _evaluate(
        lb, hd, deltas, me="Easystreet31",
        defend_buffer=payload.defend_buffer,
        scope=payload.scope,
        max_return_players=payload.max_return_players,
        players_whitelist=payload.players_whitelist
    )
    return result

@app.post("/scan_by_urls_easystreet31")
def scan_by_urls_easystreet31(payload: ScanByUrls):
    try:
        lb_raw = _fetch_table(str(payload.leaderboard_url))
        hd_raw = _fetch_table(str(payload.holdings_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch leaderboard/holdings: {e}")
    try:
        lb = _norm_lb(lb_raw)
        hd = _norm_holdings(hd_raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to normalize inputs: {e}")

    result = _scan(
        lb, hd, me="Easystreet31",
        defend_buffer=payload.defend_buffer,
        upgrade_gap=payload.upgrade_gap,
        entry_gap=payload.entry_gap,
        keep_buffer=payload.keep_buffer,
        max_each=payload.max_each,
        players_whitelist=payload.players_whitelist,
        players_exclude=payload.players_exclude
    )
    return result
