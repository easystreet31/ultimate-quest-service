import os
from app_core import app as core_app
from fastapi.middleware.cors import CORSMiddleware

# Import our routing middleware
from baseline_gets_filter import BaselineGetsFilter

app = core_app

# CORS for swagger, GPT actions, curl, etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Only add the post-routing middleware if enabled via env flag
if str(os.getenv("EVALUATE_MIDDLEWARE_REROUTE", "false")).lower() in ("1", "true", "yes", "y", "on"):
    app.add_middleware(BaselineGetsFilter)

@app.get("/info")
def info():
    return {
        "ok": True,
        "title": getattr(app, "title", "Quest Service"),
        "version": getattr(app, "version", "unknown"),
    }
