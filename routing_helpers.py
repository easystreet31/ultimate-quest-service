# routing_helpers.py
from typing import Dict, Set, Tuple
import io
import requests
import pandas as pd

LEGENDS_TAG = "LEGENDS"
ANA_TAG     = "ANA"
DAL_TAG     = "DAL"
LAK_TAG     = "LAK"
PIT_TAG     = "PIT"

DEFAULT_TAB_TO_TAG = {
    "Legends": LEGENDS_TAG,
    "ANA": ANA_TAG,
    "DAL": DAL_TAG,
    "LAK": LAK_TAG,
    "PIT": PIT_TAG,
}

def normalize_name(s: str) -> str:
    return " ".join(str(s).split()).strip()

def load_player_tags_from_xlsx(url: str,
                               tab_to_tag: Dict[str, str] = None) -> Dict[str, Set[str]]:
    """
    Reads your Google Sheet with tabs Legends/ANA/DAL/LAK/PIT and returns:
    { "Player Name": {"LEGENDS", "ANA", ...}, ... }
    """
    if tab_to_tag is None:
        tab_to_tag = DEFAULT_TAB_TO_TAG
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r.content))
    out: Dict[str, Set[str]] = {}
    for tab, tag in tab_to_tag.items():
        if tab not in xls.sheet_names:
            continue
        df = pd.read_excel(xls, sheet_name=tab)
        if df.empty:
            continue
        col0 = df.columns[0]
        for raw in df[col0].dropna().astype(str):
            name = normalize_name(raw)
            out.setdefault(name, set()).add(tag)
    return out

def choose_account_for_player(player_name: str,
                              tags_map: Dict[str, Set[str]],
                              rule_fe_just_behind: bool = True) -> str:
    """
    Rolebook routing (Section 2):
      - ANA -> UpperDuck
      - DAL/LAK/PIT -> DusterCrusher
      - Legends -> FinkleIsEinhorn (we'll keep 'just behind' in FE for now;
        overflow logic to DC is a separate score guard you already run)
      - Otherwise -> Easystreet31
    """
    name = normalize_name(player_name)
    tags = tags_map.get(name, set())
    if ANA_TAG in tags:
        return "UpperDuck"
    if any(t in tags for t in (DAL_TAG, LAK_TAG, PIT_TAG)):
        return "DusterCrusher"
    if LEGENDS_TAG in tags:
        # Per your instruction: keep FE just behind Wingnut84—even if that “hurts” QP.
        # The ‘just-behind’ *cap* is handled in your scoring/guard rail; routing stays to FE.
        return "FinkleIsEinhorn"
    return "Easystreet31"

def pick_account_to_give(player_name: str,
                         per_acct_holdings: Dict[str, Dict[str, int]]) -> str:
    """
    When a GIVE has no 'from' account specified, remove from the account with the most SP
    for that player—minimizes accidental fragility.
    """
    name = normalize_name(player_name)
    best_acct, best_sp = None, -1
    for acct, table in per_acct_holdings.items():
        sp = int(table.get(name, 0))
        if sp > best_sp:
            best_acct, best_sp = acct, sp
    return best_acct or "Easystreet31"
