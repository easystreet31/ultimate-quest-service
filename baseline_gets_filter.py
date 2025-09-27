# baseline_gets_filter.py — optional utility (kept minimal and syntactically clean)
from __future__ import annotations
from typing import Any, Dict, List

def collect_trade_get_players(trade: List[Dict[str, Any]]) -> List[str]:
    if not trade:
        return []
    out: List[str] = []
    for line in trade:
        try:
            if str(line.get("side", "")).upper() == "GET":
                name = str(line.get("players", "")).strip()
                if name:
                    out.append(name)
        except Exception:
            continue
    # unique, stable order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq
