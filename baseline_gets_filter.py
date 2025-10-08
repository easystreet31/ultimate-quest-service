import json
import os
import math
from typing import Any, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


def _is_truthy(val: str) -> bool:
    return str(val or "").lower() in ("1", "true", "yes", "y", "on")


def _looks_like_json(buf: bytes) -> bool:
    if not buf:
        return False
    s = buf.lstrip()
    return s.startswith(b"{") or s.startswith(b"[")


def _sanitize_numerics(obj: Any) -> Any:
    """
    Recursively replace non‑finite floats (NaN, +inf, -inf) with None
    so json.dumps(..., allow_nan=False) succeeds and yields valid JSON.
    """
    # Fast path for scalars
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    # Containers
    if isinstance(obj, list):
        return [_sanitize_numerics(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_numerics(v) for v in obj)
    if isinstance(obj, dict):
        return {k: _sanitize_numerics(v) for k, v in obj.items()}

    # Anything else (e.g., Decimal) – leave as is; json encoder will handle/raise
    return obj


class BaselineGetsFilter(BaseHTTPMiddleware):
    """
    Post-response middleware that *optionally* ensures the JSON body emitted by
    /family_evaluate_trade_by_urls is RFC‑compliant and never empty.

    Safety features:
      - Guarded by env var EVALUATE_MIDDLEWARE_REROUTE (truthy to enable).
      - Never "drains" the body without rebuilding it.
      - If any step fails, returns the original body unchanged.
      - Sanitizes NaN/+Inf/‑Inf -> null, then json.dumps(..., allow_nan=False).

    NOTE: This middleware does not change routing. It only validates & normalizes
    the JSON payload so tools like `jq` can parse it.
    """

    def __init__(self, app):
        super().__init__(app)
        self.enabled = _is_truthy(os.getenv("EVALUATE_MIDDLEWARE_REROUTE", ""))

    async def dispatch(self, request: Request, call_next):
        # Pass-through if disabled or not the evaluate route
        if not self.enabled or "/family_evaluate_trade_by_urls" not in request.url.path:
            return await call_next(request)

        # Call downstream once
        response = await call_next(request)

        try:
            # Safely capture the streamed body
            chunks = [chunk async for chunk in response.body_iterator]
            original_body = b"".join(chunks)
        except Exception as exc:
            # If we can't capture safely, return the original response object
            print(f"[middleware] could not capture body: {exc}")
            return response

        # Default: pass original through unchanged
        out_bytes = original_body

        # Detect JSON (by header or by shape) before attempting rewrite
        content_type = (response.headers.get("content-type") or "").lower()
        is_json_header = "application/json" in content_type or content_type.startswith("application/json")
        is_json_like = is_json_header or _looks_like_json(original_body)

        if is_json_like and original_body:
            try:
                payload = json.loads(original_body.decode("utf-8"))
                # Sanitize NaN/Inf -> None, then re‑encode strictly
                payload = _sanitize_numerics(payload)

                # Optional marker (remove if you don't want it)
                if isinstance(payload, dict):
                    meta: Dict[str, Any] = payload.setdefault("_middleware", {})
                    meta["json_sanitized"] = True

                out_bytes = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                is_json_header = True  # ensure we emit JSON content-type
            except Exception as exc:
                # Leave body as-is on any issue
                print(f"[middleware] JSON sanitize/encode skipped: {exc}")
                out_bytes = original_body

        # Build a fresh Response so content-length matches the new body
        headers = dict(response.headers)
        headers.pop("content-length", None)  # let Starlette compute it
        media_type = "application/json" if is_json_header else headers.get("content-type")

        return Response(
            content=out_bytes,
            status_code=response.status_code,
            headers=headers,
            media_type=media_type,
        )
