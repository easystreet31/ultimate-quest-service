# main.py — thin loader + optional baseline middleware
# Version: 4.1.0-unify-actions

from __future__ import annotations
import os
from fastapi import FastAPI

# --- import your real app (all routes live in app_core.py) ---
_core_err = None
try:
    from app_core import app as core_app  # <-- keep all domain logic in app_core
except Exception as e:
    _core_err = e
    core_app = None

if core_app is None:
    raise RuntimeError(f"Could not import app_core.app: {_core_err!r}")

app: FastAPI = core_app

# --- add optional middleware that helps keep fragility = traded players only ---
# If the module isn’t present or raises, the app still runs.
try:
    from baseline_gets_filter import BaselineGetsFilterMiddleware
    app.add_middleware(BaselineGetsFilterMiddleware)
except Exception:
    pass
