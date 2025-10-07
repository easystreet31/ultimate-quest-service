from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class BaselineGetsFilter(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Reserved for future: block accidental GETs to heavy routes, add request IDs, etc.
        return await call_next(request)
