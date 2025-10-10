# config.py
"""
Centralized configuration for Ultimate Quest Service.
All constants, defaults, and environment mappings in one place.
"""

import os
from typing import Dict, Set

# ============================================================================
# API / App Metadata
# ============================================================================

APP_VERSION = "5.0.0"
APP_TITLE = "Ultimate Quest Service"
APP_DESCRIPTION = "Evaluate trades, compute QP, manage family holdings for Upper Deck Ultimate Quest"

# ============================================================================
# Contest Scoring (5-3-1 format)
# ============================================================================

QP_MAP: Dict[int, int] = {
    1: 5,  # Rank 1 = 5 points
    2: 3,  # Rank 2 = 3 points
    3: 1,  # Rank 3 = 1 point
}

# ============================================================================
# Family Accounts & Routing
# ============================================================================

FAMILY_ACCOUNTS = [
    "Easystreet31",
    "DusterCrusher",
    "FinkleIsEinhorn",
    "UpperDuck",
]

# Player tag categories
PLAYER_TAG_CATEGORIES = ["LEGENDS", "ANA", "DAL", "LAK", "PIT"]

# Routing rules: which account gets priority for which tag
ROUTING_PRIORITY: Dict[str, list] = {
    "LEGENDS": ["FinkleIsEinhorn", "DusterCrusher", "Easystreet31", "UpperDuck"],
    "ANA": ["UpperDuck", "Easystreet31", "DusterCrusher", "FinkleIsEinhorn"],
    "DAL": ["DusterCrusher", "Easystreet31", "FinkleIsEinhorn", "UpperDuck"],
    "LAK": ["DusterCrusher", "Easystreet31", "FinkleIsEinhorn", "UpperDuck"],
    "PIT": ["DusterCrusher", "Easystreet31", "FinkleIsEinhorn", "UpperDuck"],
}

DEFAULT_ROUTING_ORDER = ["Easystreet31", "DusterCrusher", "FinkleIsEinhorn", "UpperDuck"]

# ============================================================================
# Rivals & Defense
# ============================================================================

SYNDICATE: Set[str] = set(
    (os.getenv("DEFAULT_TARGET_RIVALS") or "chfkyle,VjV5,FireRanger,Tfunite,Ovi8")
    .lower()
    .split(",")
)

PRIMARY_DEFENSE_BUFFER = 30  # +30 vs Syndicate (Top 25 combined)
SECONDARY_DEFENSE_BUFFER = 15  # +15 vs all other rivals

# ============================================================================
# HTTP & External Services
# ============================================================================

REQUEST_TIMEOUT_SECONDS = 15  # Reduced from 45 for better reliability
REQUEST_RETRY_MAX_ATTEMPTS = 3
REQUEST_RETRY_BACKOFF_FACTOR = 2  # exponential: 5s, 10s, 20s
REQUEST_RETRY_INITIAL_WAIT = 5
REQUESTS_USER_AGENT = "ultimate-quest-service/5.0"

# ============================================================================
# Caching
# ============================================================================

CACHE_TTL_SECONDS = 300  # 5-minute cache for sheet data
CACHE_MAX_SIZE = 64  # Max number of cached URLs

# ============================================================================
# Data Export & API Limits
# ============================================================================

MAX_DELTA_JSON_ROWS = 1000  # Cap for JSON response
DELTA_PAGINATION_DEFAULT_LIMIT = 100
DELTA_PAGINATION_MAX_LIMIT = 500

# ============================================================================
# Validation Constraints
# ============================================================================

MIN_SP_PER_TRADE = 1  # Minimum SP in any trade line
MAX_SP_PER_TRADE_LINE = 500  # Sanity check to catch data errors
MIN_DISTANCE_TO_RANK3_DEFAULT = 6  # Safe-sell minimum distance

# ============================================================================
# Environment Variable Mappings
# ============================================================================

ENV_DEFAULTS = {
    "leaderboard": "DEFAULT_LEADERBOARD_URL",
    "leaderboard_yday": "DEFAULT_LEADERBOARD_YDAY_URL",
    "holdings_e31": "DEFAULT_HOLDINGS_E31_URL",
    "holdings_dc": "DEFAULT_HOLDINGS_DC_URL",
    "holdings_fe": "DEFAULT_HOLDINGS_FE_URL",
    "holdings_ud": "DEFAULT_HOLDINGS_UD_URL",
    "collection_e31": "DEFAULT_COLLECTION_E31_URL",
    "collection_dc": "DEFAULT_COLLECTION_DC_URL",
    "collection_fe": "DEFAULT_COLLECTION_FE_URL",
    "collection_ud": "DEFAULT_COLLECTION_UD_URL",
    "pool_collection": "DEFAULT_POOL_COLLECTION_URL",
    "player_tags": "PLAYER_TAGS_URL",
}

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# Feature Flags
# ============================================================================

FORCE_FAMILY_URLS = os.getenv("FORCE_FAMILY_URLS", "true").lower() in ("1", "true", "yes")
ENABLE_WINGNUT_GUARD = True  # FE must stay behind Wingnut84 if in Top 3


def _is_truthy(val: str) -> bool:
    """Check if an env var is truthy."""
    return str(val or "").lower() in ("1", "true", "yes", "y", "on")


def get_env_url(kind: str, default: str = None) -> str:
    """Retrieve environment variable for a given kind of URL."""
    env_key = ENV_DEFAULTS.get(kind)
    if not env_key:
        raise ValueError(f"Unknown URL kind: {kind}")
    value = os.getenv(env_key, default)
    if not value:
        raise ValueError(f"Missing environment variable: {env_key}")
    return value
