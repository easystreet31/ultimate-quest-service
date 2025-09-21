from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import pandas as pd
import io

app = FastAPI(title="Ultimate Quest Service", version="1.0.0")

# -------- Helpers --------

def read_excel_anysheet(file_bytes: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    for name in ["Sheet1", "subject_leaderboards.csv"]:
        if name in xls.sheet_names:
            return pd.read_excel(xls, sheet_name=name)
    return pd.read_excel(xls, sheet_name=xls.sheet_names[0])

def build_holdings_sp(df: pd.DataFrame) -> Dict[str, float]:
    # Holdings files: columns Player | SP | QP | Rank; SP is baseline truth.
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if not set(["Player", "SP"]).issubset(df.columns):
        raise ValueError("Holdings file must contain columns: Player, SP")
    return df.groupby("Player", as_index=True)["SP"].sum().to_dict()

def parse_trade_rows(trade_df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Trade file: Player, SP, Side (GET/GIVE), optional Qty
    trade_df = trade_df.copy()
    trade_df.columns = [str(c).strip() for c in trade_df.columns]
    required = {"Player", "SP", "Side"}
    if not required.issubset(trade_df.columns):
        raise ValueError(f"Trade file must contain columns: {required}")
    rows = []
    for _, r in trade_df.iterrows():
        players = [p.strip() for p in str(r["Player"]).split("/") if p and str(p).strip()]
        sp = float(r["SP"])
        side = str(r["Side"]).strip().upper()
        qty = r["Qty"] if "Qty" in trade_df.columns else 1
        try:
            qty = int(qty)
        except:
            qty = 1
        if side not in {"GET", "GIVE"}:
            raise ValueError("Trade 'Side' must be GET or GIVE")
        rows.append({"players": players, "sp": sp, "side": side, "qty": qty})
    return rows

def leaderboard_cutoffs(leaderboard_df: pd.DataFrame) -> Dict[str, Dict[int, float]]:
    # Leaderboard: columns player | rank | sp (top-5 per player)
    df = leaderboard_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {}
    for name in df.columns:
        if name in ["player", "subject", "name"]:
            rename_map[name] = "player"
        if name in ["rank", "position"]:
            rename_map[name] = "rank"
        if name in ["sp", "score"]:
            rename_map[name] = "sp"
    df = df.rename(columns=rename_map)
    if not {"player", "rank", "sp"}.issubset(df.columns):
        raise ValueError("Leaderboard must contain columns: player, rank, sp")
    cutoffs = {}
    for player, g in df.groupby("player"):
        mapping = {}
        for _, row in g[["rank", "sp"]].dropna().iterrows():
            try:
                r = int(row["rank"]); mapping[r] = float(row["sp"])
            except:
                pass
        cutoffs[player] = dict(sorted(mapping.items(), key=lambda kv: kv[0]))
    return cutoffs

def rank_from_sp(player: str, sp: float, cuts: Dict[str, Dict[int, float]]) -> int:
    if player not in cuts:
        return 99
    best = 99
    for r, cutoff in sorted(cuts[player].items()):
        if sp >= cutoff:
            best = min(best, r)
    return best

def qp_from_rank(rank: int) -> int:
    return {1: 5, 2: 3, 3: 1}.get(rank, 0)

def apply_trade_to_sp_map(sp_map: Dict[str, float], trade_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    out = dict(sp_map)
    for row in trade_rows:
        delta = row["sp"] * row["qty"]
        for p in row["players"]:
            if row["side"] == "GET":
                out[p] = out.get(p, 0.0) + delta
            else:
                out[p] = out.get(p, 0.0) - delta
    return out

def impacted_players(trade_rows: List[Dict[str, Any]]) -> List[str]:
    s = set()
    for row in trade_rows:
        for p in row["players"]:
            s.add(p)
    return sorted(s)

# -------- Endpoint --------

@app.post("/evaluate_trade")
async def evaluate_trade(
    holdings_easystreet31: UploadFile = File(...),
    holdings_finkle: UploadFile = File(...),
    holdings_duster: UploadFile = File(...),
    leaderboard: UploadFile = File(...),
    the_collection: UploadFile = File(...),
    trade_file: UploadFile = File(...),
):
    he_bytes = await holdings_easystreet31.read()
    hf_bytes = await holdings_finkle.read()
    hd_bytes = await holdings_duster.read()
    lb_bytes = await leaderboard.read()
    tc_bytes = await the_collection.read()
    tr_bytes = await trade_file.read()

    he_df = read_excel_anysheet(he_bytes)
    hf_df = read_excel_anysheet(hf_bytes)
    hd_df = read_excel_anysheet(hd_bytes)
    lb_df = read_excel_anysheet(lb_bytes)
    tr_df = read_excel_anysheet(tr_bytes)

    he_sp = build_holdings_sp(he_df)
    hf_sp = build_holdings_sp(hf_df)
    hd_sp = build_holdings_sp(hd_df)

    trows = parse_trade_rows(tr_df)
    cuts = leaderboard_cutoffs(lb_df)

    accounts = {"Easystreet31": he_sp, "FinkleIsEinhorn": hf_sp, "DusterCrusher": hd_sp}
    players = impacted_players(trows)
    per_account = []
    overall_qp_delta = 0

    for name, sp_map in accounts.items():
        before = sp_map
        after = apply_trade_to_sp_map(sp_map, trows)
        lines = []
        acc_qp = 0
        acc_sp = 0
        for p in players:
            b_sp = float(before.get(p, 0.0))
            a_sp = float(after.get(p, 0.0))
            b_rank = rank_from_sp(p, b_sp, cuts)
            a_rank = rank_from_sp(p, a_sp, cuts)
            b_qp = qp_from_rank(b_rank)
            a_qp = qp_from_rank(a_rank)
            lines.append({
                "player": p,
                "before_sp": b_sp, "after_sp": a_sp, "delta_sp": a_sp - b_sp,
                "before_rank": b_rank, "after_rank": a_rank,
                "before_qp": b_qp, "after_qp": a_qp, "delta_qp": a_qp - b_qp
            })
            acc_qp += (a_qp - b_qp)
            acc_sp += (a_sp - b_sp)

        # Fragility: only affected players at Rank 1 or 2 with cushion < 10
        fragile = []
        for line in lines:
            if line["after_rank"] in (1, 2):
                r = line["after_rank"]
                cutoff = cuts.get(line["player"], {}).get(r, None)
                if cutoff is not None:
                    cushion = line["after_sp"] - cutoff
                    if cushion < 10:
                        fragile.append(f'{line["player"]} (SP={int(line["after_sp"])}, cushion={int(cushion)})')

        per_account.append({
            "account": name,
            "player_lines": lines,
            "net_delta_sp": acc_sp,
            "net_delta_qp": acc_qp,
            "fragile": fragile
        })
        overall_qp_delta += acc_qp

    return JSONResponse({
        "session_state": {
            "holdings": [holdings_easystreet31.filename, holdings_finkle.filename, holdings_duster.filename],
            "leaderboard": leaderboard.filename,
            "the_collection": the_collection.filename,
            "trade": trade_file.filename
        },
        "per_account": per_account,
        "overall_qp_summary": overall_qp_delta
    })
