import os
from app_core import app as core_app
from fastapi.middleware.cors import CORSMiddleware
from baseline_gets_filter import BaselineGetsFilter

app = core_app

# CORS for swagger, curl, etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Safe JSON middleware (sanitizes NaN/Inf -> null, fixes content-length)
# It runs only when EVALUATE_MIDDLEWARE_REROUTE is truthy.
app.add_middleware(BaselineGetsFilter)

@app.get("/info")
def info():
    return {
        "ok": True,
        "title": getattr(app, "title", "Quest Service"),
        "version": getattr(app, "version", "unknown"),
        "middleware_reroute_enabled": str(os.getenv("EVALUATE_MIDDLEWARE_REROUTE", "")).lower() in ("1","true","yes","y","on"),
    }
