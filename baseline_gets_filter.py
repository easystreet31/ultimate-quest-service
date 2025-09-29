# baseline_gets_filter.py — placeholder middleware (no-op)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class BaselineGetsFilter(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Keep as pass-through; useful if you later want to hard-limit GETs or add logging.
        return await call_next(request)
