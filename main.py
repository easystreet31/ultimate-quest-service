# main.py — thin wrapper around app_core with a trade-aware filter middleware
# Version: 4.0.1-tradefrag-hard + baseline-gets filter hook

from __future__ import annotations

# --- standard imports
from typing import Any
from fastapi import FastAPI

# --- import your real application (all routes live there)
# NOTE: do not change this; your service logic still resides in app_core.py
_core_err: Exception | None = None
try:
    from app_core import app as core_app
except Exception as e:
    _core_err = e
    core_app = None

if core_app is None:
    # Fail fast with a clear error if app_core doesn't load
    raise RuntimeError(f"Could not import app_core.app: {_core_err!r}")

# --- import the middleware (defined in baseline_gets_filter.py, see below)
# This middleware is a safe no-op unless the combined “trade + counter” endpoint is used.
from baseline_gets_filter import BaselineGetsFilterMiddleware

# --- bind the core app
app: FastAPI = core_app  # type: ignore[assignment]

# --- attach middleware AFTER app is bound
# (It will transparently pass through everything; on the combined trade+counter
#  endpoint it will suppress counter suggestions that duplicate GET lines.)
app.add_middleware(BaselineGetsFilterMiddleware)
