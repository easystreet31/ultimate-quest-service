# main.py — thin wrapper around app_core with a trade-aware filter middleware
# Version: 4.0.1-tradefrag-hard + baseline-gets filter hook

from __future__ import annotations
from typing import Optional
from fastapi import FastAPI

# ---- import the real app (your domain logic & all routes) ----
_core_err: Optional[Exception] = None
try:
    # Your existing application with all endpoints
    from app_core import app as core_app
except Exception as e:
    _core_err = e
    core_app = None  # type: ignore[assignment]

if core_app is None:
    # Fail fast if app_core couldn't load
    raise RuntimeError(f"Could not import app_core.app: {_core_err!r}")

# ---- our small response-filter middleware (defined in baseline_gets_filter.py) ----
from baseline_gets_filter import BaselineGetsFilterMiddleware

# ---- bind and augment the app ----
app: FastAPI = core_app  # type: ignore[assignment]

# Attach middleware AFTER the app is bound.
# It is a safe no-op unless the response contains a "counter" block next to a trade with GET lines.
app.add_middleware(BaselineGetsFilterMiddleware)
