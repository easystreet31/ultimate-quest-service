import json
import io
import os
import re
import unicodedata
from typing import Dict, Set, Any, Optional

import pandas as pd
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from urllib.request import urlopen, Request as UrlRequest


def _norm_name(s: str) -> str:
    """Normalize a player name for matching against tag sheets."""
    s = (s or "").strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Za-z0-9]+", "", s)
    return s.lower()


def _fetch_bytes(url: str, timeout: int = 30) -> bytes:
    req = UrlRequest(url, headers={"User-Agent": "QuestService/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _load_player_tags(tags_url: Optional[str]) -> Dict[str, Set[str]]:
    """
    Load tags workbook (XLSX) with tabs: Legends, ANA, DAL, LAK, PIT.
    Returns dict of normalized-name sets keyed by tag name (lowercased).
    """
    if not tags_url:
        return {}

    try:
        raw = _fetch_bytes(tags_url, timeout=45)
        xls = pd.ExcelFile(io.BytesIO(raw))
    except Exception as e:
        print(f"[tags] failed to read tags workbook: {e}")
        return {}

    tag_tabs = ["Legends", "ANA", "DAL", "LAK", "PIT"]
    result: Dict[str, Set[str]] = {}
    for tab in tag_tabs:
        if tab not in xls.sheet_names:
            continue
        try:
            df = xls.parse(tab)
            if df.shape[1] == 0:
                continue
            col0 = df.columns[0]
            names = []
            for v in df[col0].tolist():
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    names.append(_norm_name(s))
            result[tab.lower()] = set(names)
        except Exception as e:
            print(f"[tags] failed to parse sheet {tab}: {e}")

    return result


def _desired_account_for_player(player: str, tags: Dict[str, Set[str]]) -> Optional[str]:
    n = _norm_name(player)
    if "legends" in tags and n in tags["legends"]:
        return "FinkleIsEinhorn"
    if "ana" in tags and n in tags["ana"]:
        return "UpperDuck"
    if ("dal" in tags and n in tags["dal"]) or ("lak" in tags and n in tags["lak"]) or ("pit" in tags and n in tags["pit"]):
        return "DusterCrusher"
    return None  # default/no change


def _ensure_account_keys(d: Dict[str, Any]) -> None:
    """Make sure per-account dicts have all 4 family accounts present."""
    for acct in ["Easystreet31", "DusterCrusher", "FinkleIsEinhorn", "UpperDuck"]:
        if acct not in d:
            d[acct] = 0


def _reroute_evaluate_payload(payload: Dict[str, Any], tags: Dict[str, Set[str]], route_trace: bool = True) -> Dict[str, Any]:
    """
    Adjust /family_evaluate_trade_by_urls response:
    - For each GET player, move positive SP delta to desired account based on tags
    - Update allocation_plan 'to' account accordingly
    """
    if not isinstance(payload, dict):
        return payload

    # Index allocation_plan entries by player for convenience
    plan = payload.get("allocation_plan") or []
    plan_index = {}
    for i, ent in enumerate(plan):
        try:
            players = ent.get("players") or []
            if len(players) == 1:
                plan_index[_norm_name(players[0])] = i
        except Exception:
            pass

    changes = payload.get("player_changes") or []
    for pc in changes:
        try:
            player = pc.get("player", "")
            desired = _desired_account_for_player(player, tags)
            if not desired:
                # no tag match -> leave as-is
                continue

            # Prepare per-account maps
            before = dict(pc.get("per_account_sp_before") or {})
            after = dict(pc.get("per_account_sp_after") or {})
            delta = dict(pc.get("per_account_sp_delta") or {})
            _ensure_account_keys(before)
            _ensure_account_keys(after)
            _ensure_account_keys(delta)

            # Find where the positive delta landed
            pos_accounts = [a for a, v in delta.items() if isinstance(v, (int, float)) and v > 0]
            if not pos_accounts:
                # nothing to move
                continue
            # Assume a single positive assignment per GET line (as produced by evaluator)
            src = pos_accounts[0]
            amt = int(delta[src])

            if amt <= 0:
                continue

            if src == desired:
                # already correct account
                if route_trace:
                    idx = plan_index.get(_norm_name(player))
                    if idx is not None:
                        plan[idx]["routing_trace"] = {"matched_tag": desired, "action": "already_correct"}
                continue

            # Move amt from src -> desired
            delta[src] = int(delta.get(src, 0)) - amt
            delta[desired] = int(delta.get(desired, 0)) + amt

            after[src] = int(before.get(src, 0)) + int(delta.get(src, 0))
            after[desired] = int(before.get(desired, 0)) + int(delta.get(desired, 0))

            # Update the payload
            pc["per_account_sp_delta"] = delta
            pc["per_account_sp_after"] = after

            # Adjust allocation plan "to" field
            idx = plan_index.get(_norm_name(player))
            if idx is not None:
                plan[idx]["to"] = desired
                if route_trace:
                    plan[idx]["routing_trace"] = {
                        "matched_tag": desired,
                        "action": "reassigned_by_tag",
                        "from": src,
                        "moved_sp": amt
                    }

        except Exception as e:
            # Keep going on individual failures; log for debugging
            print(f"[reroute] failed to adjust player '{pc.get('player','?')}': {e}")

    # Write back adjusted plan
    payload["allocation_plan"] = plan
    return payload


class BaselineGetsFilter(BaseHTTPMiddleware):
    """
    Middleware that post-routes evaluate responses based on Player Tags
    so that GET assignments reflect the rolebook (Legends/ANA/DAL/LAK/PIT).
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Only adjust evaluate route JSON
        path = request.url.path
        method = request.method.upper()
        content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()

        if method == "POST" and path == "/family_evaluate_trade_by_urls" and content_type == "application/json":
            try:
                # Read request body to see if a player_tags_url was provided
                try:
                    req_body_bytes = await request.body()
                    req_json = json.loads(req_body_bytes.decode("utf-8")) if req_body_bytes else {}
                except Exception:
                    req_json = {}

                tags_url = req_json.get("player_tags_url") or os.getenv("PLAYER_TAGS_URL")
                tags = _load_player_tags(tags_url)

                # Extract original response body
                body_chunks = [chunk async for chunk in response.body_iterator]
                raw = b"".join(body_chunks)
                data = json.loads(raw.decode("utf-8"))

                # Re-route
                adjusted = _reroute_evaluate_payload(data, tags, route_trace=True)
                new_raw = json.dumps(adjusted, ensure_ascii=False).encode("utf-8")

                # Return new Response with same status and headers
                new_headers = dict(response.headers)
                new_headers["content-length"] = str(len(new_raw))
                return Response(content=new_raw,
                                status_code=response.status_code,
                                headers=new_headers,
                                media_type="application/json")
            except Exception as e:
                # On any failure, just return original response
                print(f"[middleware] evaluate reroute failed: {e}")
                return response

        # For all other routes, pass through unchanged
        return response
