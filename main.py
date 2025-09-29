# main.py — small-payload entrypoint
from app_core import app as core_app
from fastapi.middleware.cors import CORSMiddleware

# Use the FastAPI app defined in app_core
app = core_app

# Permissive CORS for GPT Actions, curl, and Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional /info helper (healthz already in app_core)
@app.get("/info")
def info():
    return {"ok": True, "title": app.title, "version": app.version}
