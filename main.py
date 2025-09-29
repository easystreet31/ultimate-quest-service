from app_core import app as core_app
from fastapi.middleware.cors import CORSMiddleware

app = core_app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/info")
def info():
    return {"ok": True, "title": app.title, "version": app.version}
