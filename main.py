# main.py — Ultimate Quest Service (E31 URL mode, no-quantity trades, trade-scope responses)
from typing import List, Literal, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, AnyUrl
import pandas as pd
import numpy as np
import io, re, zipfile

# ---- Robust URL fetch (requests if present, else urllib fallback) ----
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

app = FastAPI(title="Ultimate Quest Service (E31 URL mode)")

# -------------------- Models --------------------
class TradeRow(BaseModel):
    side: Literal["GET", "GIVE"]      # GET = you receive this card; GIVE = you send this card
    players: str                      # e.g., "Ryan O'Reilly/Filip Forsberg" (multi-subject)
    sp: float                         # subject points printed on the card
    # No quantities; one line = one card. For multiples, repeat lines.

class EvalByUrls(BaseModel):
    leaderboard_url: AnyUrl
    holdings_url: AnyUrl
    trade: List[TradeRow]
    multi_subject_rule: Literal["full_each_unique", "split_even"] = "full_each_unique"
    defend_buffer: int = 20
    # Keep responses small to avoid connector limits:
    scope: Literal["trade_only", "portfolio_union"] = "trade_only"
    max_return_players: int = 200
    players_whitelist: Optional[List[str]] = None

# -------------------- Helpers --------------------
def _normalize_gsheets_url(url: str, prefer_xlsx: bool = True) -> str:
    """Turn Google Sheets /edit|/view links into /export links."""
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
    return head[:2] == b"PK"  # XLSX is a ZIP container

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

    # CSV with encoding fallbacks
    for enc in (None, "utf-8", "utf-8-sig", "latin1"):
        try:
            buf.seek(0)
            df = pd.read_csv(buf, on_bad_lines="skip", encoding=enc) if enc else pd.read_csv(buf, on_bad_lines="skip")
            df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", na=False)]
            return df
        except Exception:
            continue
    raise HTTPException(status_code=400, detail="Unable to parse CSV after multiple encoding attempts.")

def _qp_from_rank(rank: int) -> int:
    return 5 if rank == 1 else 3 if rank == 2 else 1 if rank == 3 else 0

def _norm_lb(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize leaderboard to: player, user, sp, rank (keep top-5)."""
    d = df.copy()
    d = d.dropna(how="all")
    d = d.loc[:, ~d.columns.astype(str).str.contains("^Unnamed", na=False)]
    lower = {c.lower(): c for c in d.columns}

    player_col = lower.get("player") or lower.get("subject") or lower.get("name") or None
    user_col   = lower.get("user")   or lower.get("collector") or lower.get("username") or None
    sp_col     = lower.get("sp")     or lower.get("points") or lower.get("score") or None

    if player_col is None and "__sheet__" in d.columns:
        player_col = "__sheet__"
    if player_col is None:
        obj = [c for c in d.columns if d[c].dtype == object]
        if not obj: raise HTTPException(status_code=400, detail="Leaderboard missing a 'player' column.")
        player_col = obj[0]
    if user_col is None:
        obj = [c for c in d.columns if d[c].dtype == object and c != player_col]
        if not obj: raise HTTPException(status_code=400, detail="Leaderboard missing a 'user' column.")
        user_col = obj[0]
    if sp_col is None:
        nums = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        if not nums: raise HTTPException(status_code=400, detail="Leaderboard missing SP/points column.")
        sp_col = nums[0]

    out = d[[player_col, user_col, sp_col]].copy()
    out.columns = ["player", "user", "sp"]
    out["player"] = out["player"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out["user"]   = out["user"].astype(str).str.strip()
    out["sp"]     = pd.to_numeric(out["sp"], errors="coerce").fillna(0).astype(int)

    out = out.sort_values(["player", "sp"], ascending=[True, False])
    out["rank"] = out.groupby("player")["sp"].rank(method="first", ascending=False).astype(int)
    return out[out["rank"] <= 5][["player", "user", "sp", "rank"]]

def _norm_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize holdings to: player, sp, rank, qp (dedupe max SP per player)."""
    d = df.copy()
    d = d.dropna(how="all")
    d = d.loc[:, ~d.columns.astype(str).str.contains("^Unnamed", na=False)]
    lower = {c.lower(): c for c in d.columns}

    player_col = lower.get("player") or lower.get("subject") or lower.get("name") or None
    sp_col     = lower.get("sp") or lower.get("points") or lower.get("subject points") or None
    rank_col   = lower.get("rank") or lower.get("#") or lower.get("place") or None
    qp_col     = lower.get("qp") or None

    if player_col is None:
        obj = [c for c in d.columns if d[c].dtype == object]
        if not obj: raise HTTPException(status_code=400, detail="Holdings missing a 'player' column.")
        player_col = obj[0]
    if sp_col is None:
        nums = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        if not nums: raise HTTPException(status_code=400, detail="Holdings missing an SP column.")
        sp_col = nums[0]

    out = d[[player_col, sp_col]].copy()
    out.columns = ["player", "sp"]
    out["player"] = out["player"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out["sp"]     = pd.to_numeric(out["sp"], errors="coerce").fillna(0).astype(int)

    out["rank"] = pd.to_numeric(d.get(rank_col, 99), errors="coerce").fillna(99).astype(int) if rank_col in (d.columns if rank_col else []) else 99
    out["qp"]   = pd.to_numeric(d.get(qp_col, out["rank"].map({1:5,2:3,3:1})), errors="coerce").fillna(0).astype(int) if qp_col in (d.columns if qp_col else []) else out["rank"].map({1:5,2:3,3:1}).fillna(0).astype(int)

    out = out.sort_values(["player", "sp"], ascending=[True, False]).drop_duplicates("player", keep="first")
    return out[["player", "sp", "rank", "qp"]]

def _trade_to_deltas(rows: List[TradeRow], rule: str = "full_each_unique") -> Dict[str, float]:
    """
    Convert trade rows into {player -> delta SP}.
    - One row = one card (qty = 1).
    - "full_each_unique": each unique listed player receives FULL SP per card (same-player quad counts once).
    - "split_even": SP split evenly across listed players.
    """
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
            agg[n] = agg.get(n, 0.0) + sign * per_player_sp  # qty=1
    return agg

def _rank_with_me(lb: pd.DataFrame, player: str, me: str, my_sp: int):
    sub = lb[lb["player"] == player][["user", "sp"]].copy()
    if sub.empty:
        sub = pd.DataFrame([{"user": me, "sp": int(my_sp)}])
    else:
        mask_me = sub["user"].str.lower().eq(me.lower())
        if mask_me.any():
            sub.loc[mask_me, "sp"] = int(my_sp)
        else:
            sub = pd.concat([sub, pd.DataFrame([{"user": me, "sp": int(my_sp)}])], ignore_index=True)
    sub = sub.sort_values("sp", ascending=False).reset_index(drop=True)
    my_idx = sub.index[sub["user"].str.lower() == me.lower()]
    my_rank = int(my_idx[0] + 1) if len(my_idx) else 99
    first_sp  = int(sub.iloc[0]["sp"]) if len(sub) > 0 else 0
    second_sp = int(sub.iloc[1]["sp"]) if len(sub) > 1 else 0
    third_sp  = int(sub.iloc[2]["sp"]) if len(sub) > 2 else 0
    return my_rank, first_sp, second_sp, third_sp

def _evaluate(lb_df: pd.DataFrame,
              hold_df: pd.DataFrame,
              deltas: Dict[str, float],
              me: str,
              defend_buffer: int,
              scope: str,
              max_return_players: int,
              players_whitelist: Optional[List[str]]) -> Dict[str, Any]:

    delta_keys = set(deltas.keys())
    if scope == "trade_only":
        players_all = sorted(delta_keys)  # small: only trade-affected players
    else:
        players_all = sorted(set(hold_df["player"].tolist()) | delta_keys)

    if players_whitelist:
        wl = {re.sub(r"\s+", " ", p).strip() for p in players_whitelist}
        players_all = [p for p in players_all if p in wl]

    before_rows, after_rows = [], []
    for p in players_all:
        you_sp_before = int(hold_df.loc[hold_df["player"] == p, "sp"].iloc[0]) if (hold_df["player"] == p).any() else 0
        r, f, s, t = _rank_with_me(lb_df, p, me, you_sp_before)
        before_rows.append({"player": p, "you_sp": you_sp_before, "rank": r, "qp": _qp_from_rank(r),
                            "first_sp": f, "second_sp": s, "third_sp": t})

        you_sp_after = int(you_sp_before + int(round(deltas.get(p, 0))))
        r2, f2, s2, t2 = _rank_with_me(lb_df, p, me, you_sp_after)
        after_rows.append({"player": p, "you_sp_after": you_sp_after, "rank_after": r2, "qp_after": _qp_from_rank(r2),
                           "first_sp": f2, "second_sp": s2, "third_sp": t2})

    before_df = pd.DataFrame(before_rows)
    after_df  = pd.DataFrame(after_rows)
    cmp = pd.merge(before_df, after_df, on="player", how="outer").fillna(0)
    cmp["qp_delta"] = cmp["qp_after"] - cmp["qp"]
    cmp["sp_delta"] = cmp["you_sp_after"] - cmp["you_sp"]

    # Risk flags
    cmp["margin_before"] = cmp.apply(lambda r: (r["you_sp"] - r["second_sp"]) if r["rank"] == 1 else None, axis=1)
    cmp["margin_after"]  = cmp.apply(lambda r: (r["you_sp_after"] - r["second_sp"]) if r["rank_after"] == 1 else None, axis=1)
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

    # Return only touched rows to keep JSON small
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
        "portfolio_qp_before": qp_total_before,   # trade-scope (used for Δ)
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

# -------------------- Routes --------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/evaluate_by_urls_easystreet31")
def evaluate_by_urls_easystreet31(payload: EvalByUrls):
    # Fetch inputs
    try:
        lb_raw = _fetch_table(str(payload.leaderboard_url))
        hd_raw = _fetch_table(str(payload.holdings_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch leaderboard/holdings: {e}")

    # Normalize
    try:
        lb = _norm_lb(lb_raw)
        hd = _norm_holdings(hd_raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to normalize inputs: {e}")

    # Trade deltas + evaluation (small scope & small response)
    deltas = _trade_to_deltas(payload.trade, rule=payload.multi_subject_rule)
    result = _evaluate(
        lb, hd, deltas, me="Easystreet31",
        defend_buffer=payload.defend_buffer,
        scope=payload.scope,
        max_return_players=payload.max_return_players,
        players_whitelist=payload.players_whitelist
    )
    return result
