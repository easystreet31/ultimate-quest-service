import json
import os
from typing import Any, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


def _is_truthy(val: str) -> bool:
    return str(val or "").lower() in ("1", "true", "yes", "y", "on")


class BaselineGetsFilter(BaseHTTPMiddleware):
    """
    Post-response middleware that *optionally* touches the JSON payload of
    /family_evaluate_trade_by_urls.

    Safety features:
      - Fully guarded by env var EVALUATE_MIDDLEWARE_REROUTE (default: disabled).
      - Never "drains" the body and returns an empty response.
      - If any step fails, returns the original body unchanged.
      - When emitting JSON, uses allow_nan=False to guarantee valid JSON.

    NOTE: This middleware does not perform routing changes by itself; it's
    designed to be enabled only if you explicitly want it to post-process
    the evaluate endpoint output. The backend should already be authoritative
    for routing.
    """

    def __init__(self, app):
        super().__init__(app)
        self.enabled = _is_truthy(os.getenv("EVALUATE_MIDDLEWARE_REROUTE", ""))

    async def dispatch(self, request: Request, call_next):
        # Pass-through if disabled or the route isn't /family_evaluate_trade_by_urls
        if not self.enabled or "/family_evaluate_trade_by_urls" not in request.url.path:
            return await call_next(request)

        # Call the downstream handler
        response = await call_next(request)

        try:
            # Collect the raw body from the streaming iterator (without losing it)
            body_chunks = [chunk async for chunk in response.body_iterator]
            original_body = b"".join(body_chunks)
        except Exception as exc:
            # If we can't read the iterator safely, just return the original response
            # (This avoids empty bodies reaching the client.)
            print(f"[middleware] could not capture body: {exc}")
            return response

        # Default behavior: keep the original body
        out_bytes = original_body

        # Only attempt to adjust if the response is JSON
        content_type = (response.headers.get("content-type") or "").lower()
        is_json = "application/json" in content_type or content_type.startswith("application/json")
        if is_json and original_body:
            try:
                payload = json.loads(original_body.decode("utf-8"))
                # --- optional, no-op "proof of life" marker (safe to remove) ---
                if isinstance(payload, dict):
                    meta: Dict[str, Any] = payload.setdefault("_middleware", {})
                    meta["baseline_gets_filter"] = {"active": True}
                # ----------------------------------------------------------------
                out_bytes = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            except Exception as exc:
                # Leave body as-is on any parse/encode issue
                print(f"[middleware] JSON parse/encode skipped: {exc}")
                out_bytes = original_body

        # Build a fresh Response with correct content-length and content-type
        headers = dict(response.headers)
        headers.pop("content-length", None)  # let Starlette compute it
        media_type = "application/json" if is_json else headers.get("content-type")

        return Response(
            content=out_bytes,
            status_code=response.status_code,
            headers=headers,
            media_type=media_type,
        )
