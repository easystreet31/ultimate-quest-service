from app_core import app as core_app
from fastapi.middleware.cors import CORSMiddleware

app = core_app

# Wide-open CORS so Swagger, GPT Actions, and curl work from anywhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/info")
def info():
    return {"ok": True, "title": getattr(app, "title", "Quest Service"), "version": getattr(app, "version", "unknown")}
