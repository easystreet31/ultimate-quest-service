# routing_helpers.py (v5.0)
"""
Player routing logic for Ultimate Quest.
Determines which family account should hold/receive players based on tags.
Shares normalization functions with app_core for consistency.
"""

from typing import Dict, Set, Optional
import logging_utils as log_util

import config

logger = log_util.get_logger("routing_helpers")


# Import normalization from app_core to ensure consistency
# This avoids silent mismatches between tag lookups and leaderboard lookups
def _norm_key(s: str) -> str:
    """Normalize a key for lookup (shared with app_core)."""
    return " ".join(str(s or "").split()).strip().lower()


def _canon_key(s: str) -> str:
    """Canonical key (shared with app_core)."""
    raw = "".join(ch for ch in str(s or "") if ch.isalnum())
    return raw.lower()


class RoutingEngine:
    """
    Determines which family account should hold a player based on tags.
    Centralizes all routing logic and validation.
    """
    
    def __init__(self):
        self.routing_priority = config.ROUTING_PRIORITY
        self.default_routing = config.DEFAULT_ROUTING_ORDER
        self.family_accounts = config.FAMILY_ACCOUNTS
        logger.info("RoutingEngine initialized", extra={"accounts": len(self.family_accounts)})
    
    def route_player(
        self,
        player_name: str,
        player_tags: Dict[str, Set[str]],
        current_holdings: Dict[str, int],
        leaderboard_sp: Dict[str, int],
        preserve_top3: bool = True
    ) -> Dict[str, any]:
        """
        Determine best account for a player.
        
        Args:
            player_name: Display name of player
            player_tags: Dict mapping tag categories to player key sets
            current_holdings: Dict {account -> sp owned}
            leaderboard_sp: Dict {account -> current sp on leaderboard}
            preserve_top3: if True, don't move players from Top 3 accounts
        
        Returns:
            Dict with:
              - recommended_account: best account
              - reason: explanation of choice
              - all_options: ranked list of accounts
              - warnings: any caveats
        """
        player_key = _norm_key(player_name)
        warnings: list = []
        
        # Detect tags
        detected_tags = []
        for tag_category, player_keys in player_tags.items():
            if player_key in player_keys:
                detected_tags.append(tag_category)
                logger.debug(
                    f"Player tagged",
                    extra={"player": player_name, "tag": tag_category}
                )
        
        # Determine routing order
        if detected_tags:
            tag = detected_tags[0]  # Primary tag
            routing_order = self.routing_priority.get(tag, self.default_routing)
            reason = f"tag '{tag}'"
        else:
            routing_order = self.default_routing
            reason = "default routing"
        
        logger.debug(
            "Routing player",
            extra={
                "player": player_name,
                "tags": detected_tags,
                "routing_order": routing_order
            }
        )
        
        # Score each account
        scored: list = []
        for account in routing_order:
            score = 0
            
            # Priority: first in routing order
            score += 1000 - (routing_order.index(account) * 100)
            
            # Current holdings
            current_sp = current_holdings.get(account, 0)
            score += current_sp * 10
            
            # Leaderboard presence
            lb_sp = leaderboard_sp.get(account, 0)
            if lb_sp > 0:
                score += lb_sp
            
            scored.append((account, score, current_sp, lb_sp))
        
        # Sort by score
        scored.sort(key=lambda x: -x[1])
        
        recommended_account = scored[0][0]
        
        # Build result
        result = {
            "player_name": player_name,
            "player_key": player_key,
            "detected_tags": detected_tags,
            "recommended_account": recommended_account,
            "reason": reason,
            "all_options": [
                {
                    "account": acc,
                    "score": score,
                    "current_sp": curr_sp,
                    "leaderboard_sp": lb_sp
                }
                for acc, score, curr_sp, lb_sp in scored
            ],
            "warnings": warnings,
        }
        
        logger.info(
            "Player routed",
            extra={
                "player": player_name,
                "account": recommended_account,
                "tags": detected_tags
            }
        )
        
        return result
    
    def validate_routing_plan(
        self,
        get_players: Dict[str, int],  # player_key -> sp to add
        give_players: Dict[str, int],  # player_key -> sp to remove
        current_holdings: Dict[str, Dict[str, int]],  # account -> {player -> sp}
    ) -> Dict[str, any]:
        """
        Validate a complete routing plan (GETs and GIVEs).
        
        Returns:
            Dict with:
              - valid: bool
              - issues: list of problems found
              - warnings: list of cautions
        """
        issues = []
        warnings = []
        
        logger.info(
            "Validating routing plan",
            extra={
                "gets": len(get_players),
                "gives": len(give_players),
                "accounts": len(current_holdings)
            }
        )
        
        # Check GIVE feasibility
        for player_key, give_sp in give_players.items():
            total_owned = sum(
                current_holdings.get(acc, {}).get(player_key, 0)
                for acc in self.family_accounts
            )
            if total_owned < give_sp:
                issues.append(
                    f"GIVE: insufficient holdings for {player_key} "
                    f"(have {total_owned}, want to give {give_sp})"
                )
                logger.warning(
                    "Insufficient holdings for GIVE",
                    extra={"player": player_key, "have": total_owned, "want": give_sp}
                )
        
        # Check for duplicate players
        all_players = set(get_players.keys()) & set(give_players.keys())
        if all_players:
            warnings.append(
                f"Players appear in both GET and GIVE: {all_players}. "
                "This is unusual—consider reviewing."
            )
            logger.warning(
                "Duplicate GET/GIVE players",
                extra={"players": list(all_players)}
            )
        
        valid = len(issues) == 0
        
        return {
            "valid": valid,
            "issues": issues,
            "warnings": warnings,
        }


# Convenience functions for backward compatibility

_engine = RoutingEngine()


def route_player(
    player_name: str,
    tags_map: Dict[str, Set[str]],
    current_holdings: Optional[Dict[str, int]] = None,
    leaderboard_sp: Optional[Dict[str, int]] = None
) -> str:
    """
    Simple routing function: returns recommended account name.
    
    This is the legacy interface. New code should use RoutingEngine directly.
    """
    if current_holdings is None:
        current_holdings = {}
    if leaderboard_sp is None:
        leaderboard_sp = {}
    
    result = _engine.route_player(
        player_name=player_name,
        player_tags=tags_map,
        current_holdings=current_holdings,
        leaderboard_sp=leaderboard_sp
    )
    
    return result["recommended_account"]


def validate_trade(
    get_players: Dict[str, int],
    give_players: Dict[str, int],
    current_holdings: Dict[str, Dict[str, int]]
) -> bool:
    """
    Simple validation: returns True if trade is feasible.
    
    This is the legacy interface. New code should use RoutingEngine.validate_routing_plan.
    """
    result = _engine.validate_routing_plan(get_players, give_players, current_holdings)
    return result["valid"]
