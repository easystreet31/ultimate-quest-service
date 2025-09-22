# main.py  (add or replace with the following sections)
from typing import List, Literal, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, AnyUrl
import pandas as pd
import numpy as np
import io, requests

app = FastAPI(title="Ultimate Quest Service (E31 URL mode)")

# ---------- Models ----------
class TradeRow(BaseModel):
    side: Literal["GET", "GIVE"]               # GET = you receive (use Their Qty); GIVE = you send (use My Qty)
    players: str                                # e.g. "Ryan O'Reilly/Filip Forsberg"
    sp: float                                   # subject points printed on the card
    my_qty: int = 0
    their_qty: int = 0

class EvalByUrls(BaseModel):
    leaderboard_url: AnyUrl                     # CSV or XLSX URL; should contain top-5 per player
    holdings_url: AnyUrl                        # CSV or XLSX URL for Easystreet31 only
    trade: List[TradeRow]                       # a few rows at most
    multi_subject_rule: Literal["full_each_unique", "split_even"] = "full_each_unique"
    defend_buffer: int = 20                     # threshold for 'thin lead' detection

# ---------- Helpers ----------
def _load_table_from_url(url: str) -> pd.DataFrame:
    """Fetch CSV or XLSX from a URL."""
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    by = io.BytesIO(r.content)
    url_l = url.lower()
    if url_l.endswith((".xlsx", ".xls")):
        return pd.read_excel(by, engine="openpyxl")
    return pd.read_csv(by)

def _qp_from_rank(rank: int) -> int:
    return 5 if rank == 1 else 3 if rank == 2 else 1 if rank == 3 else 0

def _normalize_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Return standardized columns: player,user,sp,rank (top-5 per player)."""
    d = df.copy()
    # basic column detection
    lower = {c.lower(): c for c in d.columns}
    player_col = lower.get("player") or lower.get("subject") or lower.get("name") or None
    user_col   = lower.get("user") or lower.get("collector") or lower.get("username") or None
    sp_col     = lower.get("sp") or lower.get("points") or lower.get("score") or None
    rank_col   = lower.get("rank") or lower.get("#") or lower.get("place") or None

    # if no explicit player col, assume one sheet format—require a 'player' column for URL mode
    if player_col is None:
        if "player" in d.columns:
            player_col = "player"
        else:
            # attempt to find the most constant text column as 'player'
            obj_cols = [c for c in d.columns if d[c].dtype == object]
            if not obj_cols:
                raise HTTPException(status_code=400, detail="Leaderboard is missing a 'player' column.")
            player_col = obj_cols[0]

    if user_col is None:
        # choose first object column not the player
        obj_cols = [c for c in d.columns if d[c].dtype == object and c != player_col]
        if not obj_cols:
            raise HTTPException(status_code=400, detail="Leaderboard is missing a 'user' column.")
        user_col = obj_cols[0]

    if sp_col is None:
        num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        if not num_cols:
            raise HTTPException(status_code=400, detail="Leaderboard is missing an SP/points column.")
        sp_col = num_cols[0]

    out = d[[player_col, user_col, sp_col]].copy()
    out.columns = ["player", "user", "sp"]
    out["player"] = out["player"].astype(str).str.strip()
    out["user"]   = out["user"].astype(str).str.strip()
    out["sp"]     = pd.to_numeric(out["sp"], errors="coerce").fillna(0).astype(int)

    # Rank within player and keep top-5
    out = out.sort_values(["player", "sp"], ascending=[True, False])
    out["rank"] = out.groupby("player")["sp"].rank(method="first", ascending=False).astype(int)
    out = out[out["rank"] <= 5][["player", "user", "sp", "rank"]]
    return out

def _normalize_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Return standardized columns: player, sp, rank, qp for Easystreet31."""
    d = df.copy()
    lower = {c.lower(): c for c in d.columns}
    player_col = lower.get("player") or lower.get("subject") or lower.get("name") or None
    sp_col     = lower.get("sp") or lower.get("points") or lower.get("subject points") or None
    rank_col   = lower.get("rank") or lower.get("#") or lower.get("place") or None
    qp_col     = lower.get("qp") or None

    if player_col is None:
        # choose first object column
        obj_cols = [c for c in d.columns if d[c].dtype == object]
        if not obj_cols:
            raise HTTPException(status_code=400, detail="Holdings is missing a 'player' column.")
        player_col = obj_cols[0]

    if sp_col is None:
        num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        if not num_cols:
            raise HTTPException(status_code=400, detail="Holdings is missing an SP column.")
        sp_col = num_cols[0]

    out = d[[player_col, sp_col]].copy()
    out.columns = ["player", "sp"]
    out["player"] = out["player"].astype(str).str.strip()
    out["sp"]     = pd.to_numeric(out["sp"], errors="coerce").fillna(0).astype(int)

    # Add rank/qp if present
    if rank_col and rank_col in d.columns:
        out["rank"] = pd.to_numeric(d[rank_col], errors="coerce").fillna(99).astype(int)
    else:
        out["rank"] = 99

    if qp_col and qp_col in d.columns:
        out["qp"] = pd.to_numeric(d[qp_col], errors="coerce").fillna(0).astype(int)
    else:
        out["qp"] = out["rank"].map({1:5,2:3,3:1}).fillna(0).astype(int)

    # dedupe by best SP row per player
    out = out.sort_values(["player", "sp"], ascending=[True, False]).drop_duplicates("player", keep="first")
    return out[["player", "sp", "rank", "qp"]]

def _trade_to_deltas(trade_rows: List[TradeRow], rule: str = "full_each_unique") -> Dict[str, float]:
    """
    Convert trade rows into {player -> delta SP}.
    - "full_each_unique": each unique listed player receives FULL SP per card.
      Same-player quad counts once per card.
    - "split_even": SP split evenly across listed players.
    """
    agg: Dict[str, float] = {}
    for row in trade_rows:
        players = [p.strip() for p in row.players.split("/") if p.strip()]
        if rule == "full_each_unique":
            # de-duplicate same name (same-player quad counts once)
            players = list(dict.fromkeys(players))
            per_player_sp = float(row.sp)
        else:
            # split_even
            n = max(1, len(players))
            per_player_sp = float(row.sp) / n

        qty = row.their_qty if row.side == "GET" else row.my_qty
        sign = +1.0 if row.side == "GET" else -1.0
        for p in players:
            agg[p] = agg.get(p, 0.0) + sign * per_player_sp * float(qty)
    return agg

def _rank_with_me(leaderboard: pd.DataFrame, player: str, my_user: str, my_sp: int):
    """Merge me into the top-5 leaderboard for a player and compute my rank & top-3 SPs."""
    sub = leaderboard[leaderboard["player"] == player][["user", "sp"]].copy()
    if sub.empty:
        sub = pd.DataFrame([{"user": my_user, "sp": int(my_sp)}])
    else:
        # replace or append me
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

    # Build before/after per player
    for p in players_all:
        your_sp_before = int(holdings_df.loc[holdings_df["player"] == p, "sp"].iloc[0]) if (holdings_df["player"] == p).any() else 0
        r, f, s, t = _rank_with_me(leaderboard_df, p, my_user, your_sp_before)
        before_rows.append({"player": p, "you_sp": your_sp_before, "rank": r, "qp": _qp_from_rank(r),
                            "first_sp": f, "second_sp": s, "third_sp": t})

        your_sp_after = int(your_sp_before + int(round(deltas.get(p, 0))))
        r2, f2, s2, t2 = _rank_with_me(leaderboard_df, p, my_user, your_sp_after)
        after_rows.append({"player": p, "you_sp_after": your_sp_after, "rank_after": r2, "qp_after": _qp_from_rank(r2),
                           "first_sp": f2, "second_sp": s2, "third_sp": t2})

    before_df = pd.DataFrame(before_rows)
    after_df  = pd.DataFrame(after_rows)
    cmp = pd.merge(before_df, after_df, on="player", how="outer").fillna(0)
    cmp["qp_delta"] = cmp["qp_after"] - cmp["qp"]
    cmp["sp_delta"] = cmp["you_sp_after"] - cmp["you_sp"]

    # risk flags
    cmp["margin_before"] = cmp.apply(lambda r: (r["you_sp"] - r["second_sp"]) if r["rank"] == 1 else None, axis=1)
    cmp["margin_after"]  = cmp.apply(lambda r: (r["you_sp_after"] - r["second_sp"]) if r["rank_after"] == 1 else None, axis=1)
    cmp["created_thin_lead"] = cmp.apply(
        lambda r: 1 if (r["rank_after"] == 1 and r["margin_after"] is not None and r["margin_after"] <= defend_buffer
                        and (r["margin_before"] is None or r["margin_before"] > defend_buffer)) else 0, axis=1)
    cmp["lost_first_place"] = cmp.apply(lambda r: 1 if (r["rank"] == 1 and r["rank_after"] != 1) else 0, axis=1)

    qp_total_before = int(before_df["qp"].sum())
    qp_total_after  = int(after_df["qp_after"].sum())
    qp_delta_total  = qp_total_after - qp_total_before

    verdict = "GREEN" if (qp_delta_total > 0 and cmp["lost_first_place"].sum() == 0 and cmp["created_thin_lead"].sum() == 0) \
              else ("AMBER" if qp_delta_total >= 0 else "RED")

    # Build concise per-player list (only show changes or touched players)
    per_player = cmp.sort_values(["qp_delta", "sp_delta"], ascending=[False, False])[
        ["player", "you_sp", "you_sp_after", "sp_delta", "rank", "rank_after", "qp", "qp_after", "qp_delta",
         "margin_before", "margin_after", "created_thin_lead", "lost_first_place"]
    ].to_dict(orient="records")

    return {
        "portfolio_qp_before": qp_total_before,  # trade-scope total (used for Δ)
        "portfolio_qp_after": qp_total_after,
        "portfolio_qp_delta": qp_delta_total,
        "buffer_target": defend_buffer,
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

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/evaluate_by_urls_easystreet31")
def evaluate_by_urls_easystreet31(payload: EvalByUrls):
    try:
        lb_raw = _load_table_from_url(str(payload.leaderboard_url))
        hd_raw = _load_table_from_url(str(payload.holdings_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch leaderboard/holdings: {e}")

    try:
        lb = _normalize_leaderboard(lb_raw)
        hd = _normalize_holdings(hd_raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to normalize inputs: {e}")

    deltas = _trade_to_deltas(payload.trade, rule=payload.multi_subject_rule)
    result = _evaluate(lb, hd, deltas, my_user="Easystreet31", defend_buffer=payload.defend_buffer)
    return result
