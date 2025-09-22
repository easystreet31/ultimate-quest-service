# main.py — Ultimate Quest Service (E31 URL mode, robust Google Sheets + format sniffing)
from typing import List, Literal, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, AnyUrl
import pandas as pd
import numpy as np
import io
import re
import zipfile

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
    side: Literal["GET", "GIVE"]           # GET = receive (use Their Qty); GIVE = send (use My Qty)
    players: str                           # e.g. "Ryan O'Reilly/Filip Forsberg"
    sp: float                              # subject points on the card
    my_qty: int = 0
    their_qty: int = 0

class EvalByUrls(BaseModel):
    leaderboard_url: AnyUrl                # CSV or XLSX URL (single table or multi-sheet)
    holdings_url: AnyUrl                   # CSV or XLSX URL (Easystreet31 holdings)
    trade: List[TradeRow]
    multi_subject_rule: Literal["full_each_unique", "split_even"] = "full_each_unique"
    defend_buffer: int = 20                # thin-lead threshold

# -------------------- Helpers --------------------
def _normalize_gsheets_url(url: str, prefer_xlsx: bool = True) -> str:
    """Convert Google Sheets /edit or /view links to a direct /export link."""
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
    """Decide Excel vs CSV by URL suffix, content-type, or ZIP magic."""
    if url.lower().endswith((".xlsx", ".xls")):
        return True
    if "spreadsheetml" in content_type or "ms-excel" in content_type:
        return True
    # XLSX is a ZIP container -> starts with 'PK'
    return head[:2] == b"PK"

def _load_table_from_url(url: str) -> pd.DataFrame:
    """
    Fetch CSV/XLSX; for XLSX, concatenate all sheets with __sheet__ column.
    Robust to Google 'edit' links, odd encodings, and redirects.
    """
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
            # Sometimes Google sends CSV despite format=xlsx; fall through to CSV path
            buf.seek(0)

    # CSV path with encoding fallback
    try:
        buf.seek(0)
        df = pd.read_csv(buf, on_bad_lines="skip")
    except UnicodeDecodeError:
        buf.seek(0)
        df = pd.read_csv(buf, on_bad_lines="skip", encoding="latin1")
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", na=False)]
    return df

def _qp_from_rank(rank: int) -> int:
    return 5 if rank == 1 else 3 if rank == 2 else 1 if rank == 3 else 0

def _normalize_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to: player,user,sp,rank  (keep top-5 per player).
    If 'player' missing but __sheet__ exists (multi-tab export), use __sheet__ as player.
    """
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
        obj_cols = [c for c in d.columns if d[c].dtype == object]
        if not obj_cols:
            raise HTTPException(status_code=400, detail="Leaderboard missing a 'player' column.")
        player_col = obj_cols[0]
    if user_col is None:
        obj_cols = [c for c in d.columns if d[c].dtype == object and c != player_col]
        if not obj_cols:
            raise HTTPException(status_code=400, detail="Leaderboard missing a 'user' column.")
        user_col = obj_cols[0]
    if sp_col is None:
        num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        if not num_cols:
            raise HTTPException(status_code=400, detail="Leaderboard missing SP/points column.")
        sp_col = num_cols[0]

    out = d[[player_col, user_col, sp_col]].copy()
    out.columns = ["player", "user", "sp"]
    out["player"] = out["player"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out["user"]   = out["user"].astype(str).str.strip()
    out["sp"]     = pd.to_numeric(out["sp"], errors="coerce").fillna(0).astype(int)

    out = out.sort_values(["player", "sp"], ascending=[True, False])
    out["rank"] = out.groupby("player")["sp"].rank(method="first", ascending=False).astype(int)
    out = out[out["rank"] <= 5][["player", "user", "sp", "rank"]]
    return out

def _normalize_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize to: player, sp, rank, qp for Easystreet31. Deduplicate by max SP per player."""
    d = df.copy()
    d = d.dropna(how="all")
    d = d.loc[:, ~d.columns.astype(str).str.contains("^Unnamed", na=False)]
    lower = {c.lower(): c for c in d.columns}

    player_col = lower.get("player") or lower.get("subject") or lower.get("name") or None
    sp_col     = lower.get("sp") or lower.get("points") or lower.get("subject points") or None
    rank_col   = lower.get("rank") or lower.get("#") or lower.get("place") or None
    qp_col     = lower.get("qp") or None

    if player_col is None:
        obj_cols = [c for c in d.columns if d[c].dtype == object]
        if not obj_cols:
            raise HTTPException(status_code=400, detail="Holdings missing a 'player' column.")
        player_col = obj_cols[0]
    if sp_col is None:
        num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        if not num_cols:
            raise HTTPException(status_code=400, detail="Holdings missing an SP column.")
        sp_col = num_cols[0]

    out = d[[player_col, sp_col]].copy()
    out.columns = ["player", "sp"]
    out["player"] = out["player"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out["sp"]     = pd.to_numeric(out["sp"], errors="coerce").fillna(0).astype(int)

    if rank_col and rank_col in d.columns:
        out["rank"] = pd.to_numeric(d[rank_col], errors="coerce").fillna(99).astype(int)
    else:
        out["rank"] = 99
    if qp_col and qp_col in d.columns:
        out["qp"] = pd.to_numeric(d[qp_col], errors="coerce").fillna(0).astype(int)
    else:
        out["qp"] = out["rank"].map({1:5, 2:3, 3:1}).fillna(0).astype(int)

    out = out.sort_values(["player", "sp"], ascending=[True, False]).drop_duplicates("player", keep="first")
    return out[["player", "sp", "rank", "qp"]]

def _trade_to_deltas(trade_rows: List[TradeRow], rule: str = "full_each_unique") -> Dict[str, float]:
    """
    Convert trade rows into {player -> delta SP}.
    - "full_each_unique": each unique listed player receives FULL SP per card. Same-player quad counts once per card.
    - "split_even": SP split evenly across listed players.
    Smart qty: for GET prefer their_qty (fallback to my_qty if 0); for GIVE prefer my_qty (fallback to their_qty if 0).
    """
    agg: Dict[str, float] = {}
    for row in trade_rows:
        # Clean player string (remove newlines / double spaces)
        players_raw = re.sub(r"\s+", " ", row.players).strip()
        players = [p.strip() for p in players_raw.split("/") if p.strip()]

        if rule == "full_each_unique":
            players = list(dict.fromkeys(players))  # dedupe same-player quads
            per_player_sp = float(row.sp)
        else:
            n = max(1, len(players))
            per_player_sp = float(row.sp) / n

        if row.side == "GET":
            qty = int(row.their_qty) if row.their_qty else int(row.my_qty)
            sign = +1.0
        else:  # GIVE
            qty = int(row.my_qty) if row.my_qty else int(row.their_qty)
            sign = -1.0

        for p in players:
            agg[p] = agg.get(p, 0.0) + sign * per_player_sp * float(qty)
    return agg

def _rank_with_me(leaderboard: pd.DataFrame, player: str, my_user: str, my_sp: int):
    """Merge me into the leaderboard for a player and compute my rank & top-3 SPs."""
    sub = leaderboard[leaderboard["player"] == player][["user", "sp"]].copy()
    if sub.empty:
        sub = pd.DataFrame([{"user": my_user, "sp": int(my_sp)}])
    else:
        mask_me = sub["user"].str.lower().eq(my_user.lower())
        if mask_me.any():
            sub.loc[mask_me, "sp"] = int(my_sp)
        else:
            sub = pd.concat([sub, pd.DataFrame([{"user": my_user, "sp": int(my_sp)}])], ignore_index=True)

    sub = sub.sort_values("sp", ascending=False).reset_index(drop=True)
    my_idx = sub.index[sub["user"].str.lower() == my_user.lower()]
    my_rank = int(my_idx[0] + 1) if len(my_idx) else 99
    first_sp  = int(sub.iloc[0]["sp"]) if len(sub) > 0 else 0
    second_sp = int(sub.iloc[1]["sp"]) if len(sub) > 1 else 0
    third_sp  = int(sub.iloc[2]["sp"]) if len(sub) > 2 else 0
    return my_rank, first_sp, second_sp, third_sp

def _evaluate(leaderboard_df: pd.DataFrame,
              holdings_df: pd.DataFrame,
              deltas: Dict[str, float],
              my_user: str,
              defend_buffer: int) -> Dict[str, Any]:
    players_all = sorted(set(holdings_df["player"].tolist()) | set(deltas.keys()))
    before_rows, after_rows = [], []

    for p in players_all:
        your_sp_before = int(holdings_df.loc[holdings_df["player"] == p, "sp"].iloc[0]) if (holdings_df["player"] == p).any() else 0
        r, f, s, t = _rank_with_me(leaderboard_df, p, my_user, your_sp_before)
        before_rows.append({
            "player": p, "you_sp": your_sp_before, "rank": r, "qp": _qp_from_rank(r),
            "first_sp": f, "second_sp": s, "third_sp": t
        })

        your_sp_after = int(your_sp_before + int(round(deltas.get(p, 0))))
        r2, f2, s2, t2 = _rank_with_me(leaderboard_df, p, my_user, your_sp_after)
        after_rows.append({
            "player": p, "you_sp_after": your_sp_after, "rank_after": r2, "qp_after": _qp_from_rank(r2),
            "first_sp": f2, "second_sp": s2, "third_sp": t2
        })

    before_df = pd.DataFrame(before_rows)
    after_df  = pd.DataFrame(after_rows)
    cmp = pd.merge(before_df, after_df, on="player", how="outer").fillna(0)
    cmp["qp_delta"] = cmp["qp_after"] - cmp["qp"]
    cmp["sp_delta"] = cmp["you_sp_after"] - cmp["you_sp"]

    # Risks
    cmp["margin_before"] = cmp.apply(lambda r: (r["you_sp"] - r["second_sp"]) if r["rank"] == 1 else None, axis=1)
    cmp["margin_after"]  = cmp.apply(lambda r: (r["you_sp_after"] - r["second_sp"]) if r["rank_after"] == 1 else None, axis=1)
    cmp["created_thin_lead"] = cmp.apply(
        lambda r: 1 if (r["rank_after"] == 1 and r["margin_after"] is not None and r["margin_after"] <= defend_buffer
                        and (r["margin_before"] is None or r["margin_before"] > defend_buffer)) else 0, axis=1)
    cmp["lost_first_place"] = cmp.apply(lambda r: 1 if (r["rank"] == 1 and r["rank_after"] != 1) else 0, axis=1)

    qp_total_before = int(before_df["qp"].sum())  # trade-scope totals
    qp_total_after  = int(after_df["qp_after"].sum())
    qp_delta_total  = qp_total_after - qp_total_before

    verdict = "GREEN" if (qp_delta_total > 0 and cmp["lost_first_place"].sum() == 0 and cmp["created_thin_lead"].sum() == 0) \
              else ("AMBER" if qp_delta_total >= 0 else "RED")

    per_player = cmp.sort_values(["qp_delta", "sp_delta"], ascending=[False, False])[
        ["player","you_sp","you_sp_after","sp_delta","rank","rank_after","qp","qp_after","qp_delta",
         "margin_before","margin_after","created_thin_lead","lost_first_place"]
    ].to_dict(orient="records")

    return {
        "portfolio_qp_before": qp_total_before,
        "portfolio_qp_after":  qp_total_after,
        "portfolio_qp_delta":  qp_delta_total,
        "buffer_target":       defend_buffer,
        "risks": {
            "lost_firsts": int(cmp["lost_first_place"].sum()),
            "created_thin_leads": int(cmp["created_thin_lead"].sum()),
            "created_thin_list": [
                {"player": r["player"], "margin_after": r["margin_after"]}
                for _, r in cmp[cmp["created_thin_lead"] == 1].iterrows()
            ]
        },
        "per_player": per_player,
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
        lb_raw = _load_table_from_url(str(payload.leaderboard_url))
        hd_raw = _load_table_from_url(str(payload.holdings_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch leaderboard/holdings: {e}")

    # Normalize
    try:
        lb = _normalize_leaderboard(lb_raw)
        hd = _normalize_holdings(hd_raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to normalize inputs: {e}")

    # Trade deltas + evaluation
    deltas = _trade_to_deltas(payload.trade, rule=payload.multi_subject_rule)
    result = _evaluate(lb, hd, deltas, my_user="Easystreet31", defend_buffer=payload.defend_buffer)
    return result
