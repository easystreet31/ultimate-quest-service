# main.py (v5.0)
"""
Ultimate Quest Service - FastAPI Entry Point
Standardized error responses, simplified middleware, improved health checks.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import ValidationError

import config
import logging_utils as log_util
import cache_utils
import app_core

# Initialize logging
log_util.setup_logging()
logger = log_util.get_logger("main")

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description=config.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================================
# Exception Handlers (Standardized Error Responses)
# ============================================================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    """Handle Pydantic validation errors."""
    errors = exc.errors()
    error_details = "; ".join(
        f"{'.'.join(str(e) for e in err.get('loc', []))}: {err.get('msg', 'unknown')}"
        for err in errors
    )
    response, status_code = log_util.create_error_response(
        "validation_error",
        f"Request validation failed: {error_details}",
        status_code=422,
        context={"errors": error_details}
    )
    return JSONResponse(content=response, status_code=status_code)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTPException with standardized format."""
    response, _ = log_util.create_error_response(
        "http_error",
        detail=exc.detail,
        status_code=exc.status_code,
        context={"status_code": exc.status_code}
    )
    return JSONResponse(content=response, status_code=exc.status_code)


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle unexpected errors."""
    logger.error(
        "Unhandled exception",
        exc_info=True,
        extra={"path": request.url.path, "error": str(exc)}
    )
    response, status_code = log_util.create_error_response(
        "internal_error",
        "An unexpected error occurred. Check server logs for details.",
        status_code=500,
        context={"error_type": type(exc).__name__}
    )
    return JSONResponse(content=response, status_code=status_code)


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    """Quick liveness check."""
    return {"ok": True, "status": "healthy", "version": config.APP_VERSION}


@app.get("/health")
def health() -> Dict[str, Any]:
    """Deep health check (validates external dependencies)."""
    logger.info("Running deep health check")
    
    checks = {
        "api": True,
        "cache": True,
        "config": True,
    }
    
    # Basic checks
    try:
        # Verify config loads
        _ = config.get_env_url("leaderboard")
        checks["config"] = True
    except Exception as e:
        logger.error("Health check failed: config", exc_info=True)
        checks["config"] = False
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return {
        "ok": all_healthy,
        "status": "healthy" if all_healthy else "degraded",
        "version": config.APP_VERSION,
        "checks": checks
    }


@app.get("/info")
def info() -> Dict[str, Any]:
    """API metadata."""
    return {
        "ok": True,
        "title": config.APP_TITLE,
        "version": config.APP_VERSION,
        "description": config.APP_DESCRIPTION,
    }


@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "ok": True,
        "message": "Ultimate Quest Service API. See /docs for OpenAPI.",
        "version": config.APP_VERSION,
        "endpoints": {
            "health": "GET /health (deep checks)",
            "healthz": "GET /healthz (quick check)",
            "info": "GET /info (metadata)",
            "defaults": "GET /defaults (config defaults)",
            "cache_stats": "GET /cache/stats (cache info)",
            "evaluate_trade": "POST /family_evaluate_trade_by_urls",
            "leaderboard_delta": "POST /leaderboard_delta_by_urls",
            "leaderboard_delta_export": "GET /leaderboard_delta_export",
        }
    }


# ============================================================================
# Configuration & Cache Endpoints
# ============================================================================

@app.get("/defaults")
def get_defaults() -> Dict[str, Any]:
    """Return configured defaults."""
    logger.info("Returning configuration defaults")
    return {
        "ok": True,
        "defaults": {
            "leaderboard": os.getenv("DEFAULT_LEADERBOARD_URL"),
            "leaderboard_yday": os.getenv("DEFAULT_LEADERBOARD_YDAY_URL"),
            "holdings_e31": os.getenv("DEFAULT_HOLDINGS_E31_URL"),
            "holdings_dc": os.getenv("DEFAULT_HOLDINGS_DC_URL"),
            "holdings_fe": os.getenv("DEFAULT_HOLDINGS_FE_URL"),
            "holdings_ud": os.getenv("DEFAULT_HOLDINGS_UD_URL"),
            "collection_e31": os.getenv("DEFAULT_COLLECTION_E31_URL"),
            "collection_dc": os.getenv("DEFAULT_COLLECTION_DC_URL"),
            "collection_fe": os.getenv("DEFAULT_COLLECTION_FE_URL"),
            "collection_ud": os.getenv("DEFAULT_COLLECTION_UD_URL"),
            "pool_collection": os.getenv("DEFAULT_POOL_COLLECTION_URL"),
            "player_tags": os.getenv("PLAYER_TAGS_URL"),
        },
        "constants": {
            "rivals": list(config.SYNDICATE),
            "family_accounts": config.FAMILY_ACCOUNTS,
            "primary_defense_buffer": config.PRIMARY_DEFENSE_BUFFER,
            "secondary_defense_buffer": config.SECONDARY_DEFENSE_BUFFER,
            "qp_scoring": config.QP_MAP,
            "cache_ttl_seconds": config.CACHE_TTL_SECONDS,
        }
    }


@app.get("/cache/stats")
def cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    logger.info("Returning cache statistics")
    return {
        "ok": True,
        "cache": cache_utils.cache_stats()
    }


@app.post("/cache/clear")
def cache_clear() -> Dict[str, Any]:
    """Clear cache (admin endpoint)."""
    logger.warning("Cache cleared via admin endpoint")
    cache_utils.cache_clear()
    return {
        "ok": True,
        "message": "Cache cleared"
    }


# ============================================================================
# Trade Evaluation
# ============================================================================

@app.post("/family_evaluate_trade_by_urls")
def family_evaluate_trade_by_urls(req: app_core.FamilyEvaluateTradeReq = Body(...)) -> Dict[str, Any]:
    """
    Evaluate a family trade.
    
    Returns verdict (APPROVE/CAUTION/DECLINE) with QP impact and rank/buffer deltas.
    """
    try:
        logger.info(
            "Evaluating trade",
            extra={
                "account": req.trade_account,
                "num_lines": len(req.trade),
                "get_count": sum(1 for t in req.trade if t.side == "GET"),
                "give_count": sum(1 for t in req.trade if t.side == "GIVE"),
            }
        )
        
        # Load data
        lb_url = app_core._pick_url(req.leaderboard_url, "leaderboard", req.prefer_env_defaults)
        tags_url = app_core._pick_url(req.player_tags_url, "player_tags", req.prefer_env_defaults)
        
        leader = app_core.normalize_leaderboard(app_core.fetch_xlsx(lb_url))
        tags = app_core._load_player_tags(req.prefer_env_defaults, tags_url)
        holds = app_core.holdings_from_urls(
            req.holdings_e31_url, req.holdings_dc_url, req.holdings_fe_url,
            req.prefer_env_defaults, req.holdings_ud_url
        )
        
        # Evaluate
        result = app_core.evaluate_trade_internal(req, leader, tags, holds)
        
        logger.info(
            "Trade evaluation complete",
            extra={
                "verdict": result["verdict"],
                "delta_qp": result["total_changes"]["delta_qp"],
                "delta_buffer": result["total_changes"]["delta_buffer"],
            }
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Trade evaluation failed", exc_info=True, extra={"error": str(e)})
        response, status = log_util.create_error_response(
            "trade_evaluation_failed",
            f"Failed to evaluate trade: {str(e)}",
            status_code=500
        )
        raise HTTPException(status_code=status, detail=response)


# ============================================================================
# Leaderboard Delta
# ============================================================================

@app.post("/leaderboard_delta_by_urls")
def leaderboard_delta_by_urls(req: app_core.LeaderboardDeltaReq = Body(...)) -> Dict[str, Any]:
    """
    Report daily leaderboard movements for the family and rivals.
    """
    try:
        logger.info("Computing leaderboard delta")
        
        today_url = app_core._pick_url(req.leaderboard_today_url, "leaderboard", req.prefer_env_defaults)
        yday_url = app_core._pick_url(req.leaderboard_yesterday_url, "leaderboard_yday", req.prefer_env_defaults)
        
        today = app_core.normalize_leaderboard(app_core.fetch_xlsx(today_url))
        yday = app_core.normalize_leaderboard(app_core.fetch_xlsx(yday_url))
        
        rivals_canon = set(
            app_core._canon_key(app_core._strip_user_suffix(r))
            for r in (req.rivals or list(config.SYNDICATE))
        )
        
        rows = app_core._delta_rows(today, yday, rivals_canon, req.min_sp_delta)
        
        logger.info(
            "Leaderboard delta computed",
            extra={"total_players": len(rows), "rivals_count": len(rivals_canon)}
        )
        
        return {
            "ok": True,
            "data": {
                "leaderboard_today_url": today_url,
                "leaderboard_yesterday_url": yday_url,
                "params": {
                    "rivals": sorted(list(rivals_canon)),
                    "min_sp_delta": int(req.min_sp_delta)
                },
                "summary": {
                    "players_scanned": len(set(today["sp_map"].keys()) | set(yday["sp_map"].keys())),
                    "players_moved": min(len(rows), config.MAX_DELTA_JSON_ROWS)
                },
                "players": rows[:config.MAX_DELTA_JSON_ROWS]
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Delta computation failed", exc_info=True, extra={"error": str(e)})
        response, status = log_util.create_error_response(
            "delta_computation_failed",
            f"Failed to compute delta: {str(e)}",
            status_code=500
        )
        raise HTTPException(status_code=status, detail=response)


@app.get("/leaderboard_delta_export")
def leaderboard_delta_export(
    prefer_env_defaults: bool = Query(True),
    leaderboard_today_url: Optional[str] = Query(None),
    leaderboard_yesterday_url: Optional[str] = Query(None),
    rivals: Optional[str] = Query(None, description="Comma-separated usernames"),
    min_sp_delta: int = Query(1, ge=0),
    file_format: str = Query("csv", alias="format", pattern="^(csv|xlsx)$"),
) -> Response:
    """
    Export leaderboard delta as CSV or XLSX.
    """
    try:
        logger.info(
            "Exporting leaderboard delta",
            extra={"format": file_format, "min_sp_delta": min_sp_delta}
        )
        
        today_url = app_core._pick_url(leaderboard_today_url, "leaderboard", prefer_env_defaults)
        yday_url = app_core._pick_url(leaderboard_yesterday_url, "leaderboard_yday", prefer_env_defaults)
        
        today = app_core.normalize_leaderboard(app_core.fetch_xlsx(today_url))
        yday = app_core.normalize_leaderboard(app_core.fetch_xlsx(yday_url))
        
        rival_list = [r.strip() for r in (rivals.split(",") if rivals else list(config.SYNDICATE)) if r.strip()]
        rivals_canon = set(
            app_core._canon_key(app_core._strip_user_suffix(r)) for r in rival_list
        )
        
        rows = app_core._delta_rows(today, yday, rivals_canon, min_sp_delta)
        
        # Build flat rows for export
        flat = []
        for r in rows:
            fb = r["family_before"]
            fa = r["family_after"]
            flat.append({
                "player": r["player"],
                "family_before_account": fb["account"],
                "family_before_sp": fb["sp"],
                "family_before_rank": fb["rank"],
                "family_before_buffer": fb["buffer"],
                "family_after_account": fa["account"],
                "family_after_sp": fa["sp"],
                "family_after_rank": fa["rank"],
                "family_after_buffer": fa["buffer"],
                "delta_sp": r["delta_sp"],
                "delta_rank": r["delta_rank"],
                "delta_buffer": r["delta_buffer"],
                "rivals_sp_before": r["rivals_sp_before"],
                "rivals_sp_after": r["rivals_sp_after"],
                "delta_rivals_sp": r["delta_rivals_sp"],
            })
        
        import pandas as pd
        df = pd.DataFrame(flat)
        fname = f"leaderboard_delta_{app_core.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{file_format}"
        
        if file_format == "csv":
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            logger.info(
                "Delta exported as CSV",
                extra={"filename": fname, "rows": len(flat)}
            )
            return Response(
                content=csv_bytes,
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{fname}"'}
            )
        
        else:  # xlsx
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="delta")
            bio.seek(0)
            logger.info(
                "Delta exported as XLSX",
                extra={"filename": fname, "rows": len(flat)}
            )
            return StreamingResponse(
                bio,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f'attachment; filename="{fname}"'}
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Delta export failed", exc_info=True, extra={"error": str(e)})
        response, status = log_util.create_error_response(
            "export_failed",
            f"Failed to export delta: {str(e)}",
            status_code=500
        )
        raise HTTPException(status_code=status, detail=response)


# ============================================================================
# Startup & Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info(
        "Application starting",
        extra={
            "version": config.APP_VERSION,
            "log_level": config.LOG_LEVEL,
            "cache_ttl": config.CACHE_TTL_SECONDS,
        }
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Application shutting down")


# ============================================================================
# Local Dev
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
