# main.py — thin wrapper around app_core with counter-pick safety middleware
# Version: 4.1.0 (Small-Payload API)

from __future__ import annotations
from typing import Optional
from fastapi import FastAPI

# Import the real app (domain logic & routes)
_core_err: Optional[Exception] = None
try:
    from app_core import app as core_app
except Exception as e:
    _core_err = e
    core_app = None  # type: ignore[assignment]

if core_app is None:
    raise RuntimeError(f"Could not import app_core.app: {_core_err!r}")

# Attach middleware that prunes counter picks overlapping trade GET players
from baseline_gets_filter import BaselineGetsFilterMiddleware

app: FastAPI = core_app  # type: ignore[assignment]
app.add_middleware(BaselineGetsFilterMiddleware)
