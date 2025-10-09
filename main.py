from app_core import app as core_app
from fastapi.middleware.cors import CORSMiddleware
from baseline_gets_filter import BaselineGetsFilter

app = core_app

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Post-routing middleware for evaluate endpoint
app.add_middleware(BaselineGetsFilter)

@app.get("/info")
def info():
    return {
        "ok": True,
        "title": getattr(app, "title", "Quest Service"),
        "version": getattr(app, "version", "unknown")
    }
