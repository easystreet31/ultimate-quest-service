# --- imports you already have ---
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import io, requests, math, re
import pandas as pd

app = FastAPI()

# --------- utilities (robust to your Google links) ----------
def _normalize_gsheet_url(u: str) -> str:
    if "/export?format=" in u:
        return u
    if "/edit" in u or "/view" in u:
        return re.sub(r"/(edit|view).*", "/export?format=xlsx", u)
    if "/spreadsheets/d/" in u and "export?format=" not in u:
        if u.endswith("/"):
            return u + "export?format=xlsx"
        return u + "/export?format=xlsx"
    return u

def _fetch_xlsx(u: str) -> dict:
    u = _normalize_gsheet_url(u)
    r = requests.get(u, timeout=60)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), sheet_name=None)

def _col(df: pd.DataFrame, *cands):
    lower = {c.lower(): c for c in df.columns}
    for c in cands:
        k = c.lower()
        if k in lower:
            return lower[k]
    return None

def _players_from_text(txt: str) -> list:
    if pd.isna(txt):
        return []
    s = str(txt).replace("&", ",").replace("/", ",")
    parts = [p.strip() for p in re.split(r"[,+]| and ", s) if p.strip()]
    # de‑dup while preserving order
    seen, out = set(), []
    for p in parts:
        if p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out

# --------- data loaders ----------
def load_leaderboard(leaderboard_url: str) -> pd.DataFrame:
    wb = _fetch_xlsx(leaderboard_url)
    frames = []
    for name, df in wb.items():
        if df.empty: 
            continue
        pcol = _col(df, "player", "subject", "subject/player", "name")
        rcol = _col(df, "rank")
        ucol = _col(df, "user", "username", "collector", "owner")
        scol = _col(df, "sp", "subject points", "points")
        if not all([pcol, rcol, ucol, scol]): 
            continue
        f = df[[pcol, rcol, ucol, scol]].copy()
        f.columns = ["player", "rank", "user", "sp"]
        frames.append(f)
    if not frames:
        raise ValueError("Could not locate a leaderboard sheet with [player, rank, user, sp] columns.")
    lb = pd.concat(frames, ignore_index=True)
    # keep top 5 ranks per player
    lb = lb.dropna(subset=["player", "rank"]).copy()
    lb["rank"] = lb["rank"].astype(int)
    lb = lb[lb["rank"].between(1, 5)]
    # normalize usernames
    lb["user"] = lb["user"].astype(str).str.strip()
    return lb

def load_holdings(holdings_url: str) -> pd.DataFrame:
    wb = _fetch_xlsx(holdings_url)
    frames = []
    for name, df in wb.items():
        if df.empty:
            continue
        pcol = _col(df, "player", "subject", "name")
        scol = _col(df, "sp", "subject points", "your sp", "total sp")
        rcol = _col(df, "rank")
        qpcol= _col(df, "qp", "quest points")
        if not pcol or not scol:
            continue
        f = df[[pcol, scol] + ([rcol] if rcol else []) + ([qpcol] if qpcol else [])].copy()
        cols = ["player","you_sp"] + (["rank"] if rcol else []) + (["you_qp"] if qpcol else [])
        f.columns = cols
        frames.append(f)
    if not frames:
        raise ValueError("Could not locate holdings columns.")
    h = pd.concat(frames, ignore_index=True)
    # collapse duplicates by player
    h = h.groupby("player", as_index=False)["you_sp"].sum()
    return h

def load_collection(collection_url: str) -> pd.DataFrame:
    wb = _fetch_xlsx(collection_url)
    frames = []
    for name, df in wb.items():
        if df.empty:
            continue
        ncol = _col(df, "card", "card name", "name", "item")
        pcol = _col(df, "players", "subjects", "subject(s)")
        scol = _col(df, "sp", "subject points", "points")
        qcol = _col(df, "qty", "quantity", "count")
        if not all([ncol, pcol, scol]):
            continue
        f = df[[ncol, pcol, scol] + ([qcol] if qcol else [])].copy()
        f.columns = ["card", "players", "sp"] + (["qty"] if qcol else [])
        if "qty" not in f.columns:
            f["qty"] = 1
        frames.append(f)
    if not frames:
        raise ValueError("Could not locate collection columns [card, players, sp, qty].")
    c = pd.concat(frames, ignore_index=True)
    # parse players
    c["players_list"] = c["players"].apply(_players_from_text)
    # drop rows with no subjects parsed
    c = c[c["players_list"].map(len) > 0].copy()
    return c

# --------- leaderboard comparison utilities ----------
def build_cmp_table(lb: pd.DataFrame, you_hold: pd.DataFrame, you_name: str = "Easystreet31") -> pd.DataFrame:
    # build top5 columns wide
    wide = {}
    for player, group in lb.groupby("player"):
        r = {}
        for _, row in group.iterrows():
            rk = int(row["rank"])
            r[f"r{rk}_user"] = str(row["user"]).strip()
            r[f"r{rk}_sp"]   = int(row["sp"])
        wide[player] = r
    cmp = pd.DataFrame.from_dict(wide, orient="index").reset_index().rename(columns={"index":"player"})
    # fill missing ranks with 0
    for k in range(1,6):
        if f"r{k}_user" not in cmp.columns: cmp[f"r{k}_user"] = None
        if f"r{k}_sp" not in cmp.columns:   cmp[f"r{k}_sp"]   = 0
    # attach your SP
    cmp = cmp.merge(you_hold.rename(columns={"player":"player","you_sp":"you_sp"}), on="player", how="left")
    cmp["you_sp"] = cmp["you_sp"].fillna(0).astype(int)

    # compute your rank relative to top5 bands
    def my_rank(r):
        y = r["you_sp"]
        # compare to rank 1..5 SP (ties treated conservatively as losing)
        bands = [r[f"r{k}_sp"] for k in range(1,6)]
        if y > bands[0]: return 1
        if y > bands[1]: return 2
        if y > bands[2]: return 3
        if y > bands[3]: return 4
        if y > bands[4]: return 5
        return None  # outside top5
    cmp["you_rank"] = cmp.apply(my_rank, axis=1)

    # friendly aliases for later math
    cmp["first_sp"]  = cmp["r1_sp"].fillna(0).astype(int)
    cmp["second_sp"] = cmp["r2_sp"].fillna(0).astype(int)
    cmp["third_sp"]  = cmp["r3_sp"].fillna(0).astype(int)
    cmp["fourth_sp"] = cmp["r4_sp"].fillna(0).astype(int)
    cmp["fifth_sp"]  = cmp["r5_sp"].fillna(0).astype(int)
    return cmp

def safe_slack_for_you(row, defend_buffer: int = 20) -> int:
    y, r1, r2, r3, r4, r5, myrk = row["you_sp"], row["first_sp"], row["second_sp"], row["third_sp"], row["fourth_sp"], row["fifth_sp"], row["you_rank"]
    if myrk == 1:
        # keep at least (second + buffer + 1)
        need = (r2 + defend_buffer + 1)
        return max(0, y - need)
    if myrk == 2:
        # keep above third by 1
        need = (r3 + 1)
        return max(0, y - need)
    if myrk == 3:
        # keep above fourth by 1
        need = (r4 + 1)
        return max(0, y - need)
    # outside top3: any give is QP‑safe
    return 10**9  # effectively unlimited

def qp_band(rank_before: Optional[int]) -> int:
    if rank_before == 1: return 5
    if rank_before == 2: return 3
    if rank_before == 3: return 1
    return 0

def still_same_qp_band(row, give_sp: int, defend_buffer: int = 20) -> bool:
    """Would giving 'give_sp' keep you in the same QP band for this player?"""
    before = row["you_rank"]
    after_sp = row["you_sp"] - give_sp
    # Recompute rank coarsely vs bands (conservative ties)
    r1, r2, r3, r4, r5 = row["first_sp"], row["second_sp"], row["third_sp"], row["fourth_sp"], row["fifth_sp"]
    if before == 1:
        return after_sp > (r2 + defend_buffer)
    if before == 2:
        return after_sp > r3
    if before == 3:
        return after_sp > r4
    # outside top3 -> band=0 regardless
    return True

def partner_baseline_sp(lb_for_player: pd.DataFrame, partner: str) -> int:
    """If partner appears in top5 for this player, return their SP; otherwise 0 (conservative)."""
    m = lb_for_player[lb_for_player["user"].str.strip().str.lower() == partner.lower()]
    if not m.empty:
        return int(m.iloc[0]["sp"])
    return 0

# --------- offer planner core ----------
def compute_offers(
    leaderboard_url: str,
    holdings_url: str,
    my_collection_url: str,
    partner: str,
    *,
    defend_buffer: int = 20,
    prefer_hurt_rivals: bool = True,
    target_rivals: list = None,
    max_each: int = 0,
    max_multiples_per_card: int = 3,
    multi_subject_rule: str = "full_each_unique",
):
    lb = load_leaderboard(leaderboard_url)
    you = load_holdings(holdings_url)
    inv = load_collection(my_collection_url)
    cmp = build_cmp_table(lb, you)

    # index lb by player for quick access
    lb_by_player = {p: g.copy() for p, g in lb.groupby("player")}

    suggestions = []
    diag_errors = []

    # precompute safe slack per player
    slack = {}
    for _, r in cmp.iterrows():
        slack[r["player"]] = safe_slack_for_you(r, defend_buffer=defend_buffer)

    # Walk each inventory line and check if giving N copies can:
    #  - keep your QP band unchanged for *all* included players
    #  - put partner into Top3 or 1st for *any* included player (conservative baseline if not in top5)
    for idx, row in inv.iterrows():
        card = row["card"]
        card_sp = int(row["sp"])
        qty = int(row.get("qty", 1))
        players = list(row["players_list"])
        if not players or card_sp <= 0 or qty <= 0:
            continue

        # respect multi-subject: full SP for each unique player on the card
        # (duplicates already de‑duplicated in players_list)
        # compute max safe copies limited by slack on *every* subject
        max_by_slack = 10**9
        all_subjects_have_lb = True
        for subj in players:
            # If player not on leaderboard, we can't reason about your QP band;
            # treat it as safe (you have 0 QP there).
            row_cmp = cmp[cmp["player"] == subj]
            if row_cmp.empty:
                subj_slack = 10**9  # untracked subject: doesn't affect your QP
            else:
                subj_slack = slack.get(subj, 10**9)
                # ensure band safety at the per‑copy granularity
                # c copies consume (card_sp * c)
            max_copies_for_subj = subj_slack // card_sp
            max_by_slack = min(max_by_slack, max_copies_for_subj)

        safe_copies_cap = max(0, min(qty, max_by_slack, max_multiples_per_card))
        if safe_copies_cap <= 0:
            continue

        # For each subject on the card, see if N copies could move partner to Top3 / 1st.
        best_pick = None  # keep the best variant for this card
        for subj in players:
            lbp = lb_by_player.get(subj)
            if lbp is None or lbp.empty:
                continue
            p_sp = partner_baseline_sp(lbp, partner=partner)
            first_sp  = int(lbp[lbp["rank"]==1]["sp"].iloc[0]) if (lbp["rank"]==1).any() else 0
            third_sp  = int(lbp[lbp["rank"]==3]["sp"].iloc[0]) if (lbp["rank"]==3).any() else 0

            need_top3 = max(0, (third_sp + 1) - p_sp)
            need_first = max(0, (first_sp + 1) - p_sp)

            # how many copies of this card would be needed for subj?
            # (conservative: only this single card type; we don't mix types)
            def copies_for(target_need: int) -> int:
                if target_need <= 0:
                    return 0
                return math.ceil(target_need / card_sp)

            c3 = copies_for(need_top3)
            c1 = copies_for(need_first)

            # filter by safe copies cap
            can_top3 = (c3 > 0 and c3 <= safe_copies_cap)
            can_first= (c1 > 0 and c1 <= safe_copies_cap)

            # evaluate the QP change for YOU if you give c copies (across all subjects on the card)
            def band_safe_for_copies(copies: int) -> bool:
                give = copies * card_sp
                for s in players:
                    row_cmp = cmp[cmp["player"] == s]
                    if row_cmp.empty:
                        continue
                    if not still_same_qp_band(row_cmp.iloc[0], give_sp=give, defend_buffer=defend_buffer):
                        return False
                return True

            pick = None
            target = None
            copies_needed = None
            if can_first and band_safe_for_copies(c1):
                target = "take_1st"
                copies_needed = c1
            elif can_top3 and band_safe_for_copies(c3):
                target = "enter_top3"
                copies_needed = c3

            if target:
                # prefer hurting a named rival if displacement occurs
                hurts = None
                if prefer_hurt_rivals and target_rivals:
                    # who gets displaced?
                    if target == "enter_top3":
                        displaced = str(lbp[lbp["rank"]==3]["user"].iloc[0]) if (lbp["rank"]==3).any() else ""
                    else:
                        displaced = str(lbp[lbp["rank"]==1]["user"].iloc[0]) if (lbp["rank"]==1).any() else ""
                    for rv in target_rivals:
                        if rv and displaced.strip().lower() == rv.strip().lower():
                            hurts = rv
                            break

                score = (10 if target == "take_1st" else 4) + (3 if hurts else 0)
                pick = {
                    "card": card,
                    "players_on_card": players,
                    "subject": subj,
                    "sp_per_copy": card_sp,
                    "copies_to_offer": copies_needed,
                    "copies_available": int(qty),
                    "copies_safe_cap": int(safe_copies_cap),
                    "partner_need_sp": int(need_first if target=="take_1st" else need_top3),
                    "partner_target": target,
                    "your_qp_change": 0,   # guaranteed by band_safe
                    "hurts_rival": hurts,
                    "score": score
                }

            if pick:
                if (best_pick is None) or (pick["score"] > best_pick["score"]):
                    best_pick = pick

        if best_pick:
            suggestions.append(best_pick)

    # sort by score, then by smaller copies_to_offer
    suggestions.sort(key=lambda x: (-x["score"], x["copies_to_offer"], x["subject"], x["card"]))

    # apply max_each if requested
    if max_each and max_each > 0:
        suggestions = suggestions[:max_each]

    return {
        "params": {
            "defend_buffer": defend_buffer,
            "multi_subject_rule": multi_subject_rule,
            "partner": partner,
            "prefer_hurt_rivals": prefer_hurt_rivals,
            "target_rivals": target_rivals or [],
            "max_each": max_each,
            "max_multiples_per_card": max_multiples_per_card
        },
        "inventory_seen": {
            "card_types": int(inv.shape[0]),
            "total_units": int(inv["qty"].sum())
        },
        "safe_offers": suggestions,
        "omitted": 0 if (not max_each or max_each <= 0) else max(0, len(suggestions) - max_each),
        "diagnostics": {
            "players_in_holdings": int(you.shape[0]),
            "players_on_leaderboard": int(lb["player"].nunique()),
            "errors": diag_errors[:10]
        }
    }

# ------------- NEW ROUTE ----------------
@app.post("/offer_partner_by_urls_easystreet31")
def offer_partner_by_urls_easystreet31(payload: dict):
    """
    Body:
    {
      "leaderboard_url": "...xlsx",
      "holdings_url": "...xlsx",
      "my_collection_url": "...xlsx",
      "partner": "Samm78ca",
      "defend_buffer": 20,
      "prefer_hurt_rivals": true,
      "target_rivals": ["chfkyle","Erikk"],
      "max_each": 60,
      "max_multiples_per_card": 3,
      "multi_subject_rule": "full_each_unique"
    }
    """
    try:
        result = compute_offers(
            leaderboard_url=payload["leaderboard_url"],
            holdings_url=payload["holdings_url"],
            my_collection_url=payload["my_collection_url"],
            partner=payload.get("partner", ""),
            defend_buffer=int(payload.get("defend_buffer", 20)),
            prefer_hurt_rivals=bool(payload.get("prefer_hurt_rivals", True)),
            target_rivals=payload.get("target_rivals") or [],
            max_each=int(payload.get("max_each", 0)),
            max_multiples_per_card=int(payload.get("max_multiples_per_card", 3)),
            multi_subject_rule=payload.get("multi_subject_rule","full_each_unique"),
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"detail": f"Offer planner failed: {e}"}, status_code=500)
