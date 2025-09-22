from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import json

app = FastAPI()

@app.post("/evaluate_trade_json")
async def evaluate_trade_json(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = json.loads(contents.decode("utf-8"))

        required_keys = [
            "easystreet31",
            "finkleiseinhorn",
            "dustercrusher",
            "leaderboard",
            "the_collection",
            "trade"
        ]

        missing = [k for k in required_keys if k not in data]

        response = {
            "received_keys": list(data.keys()),
            "missing_keys": missing
        }

        if missing:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Missing required keys",
                    **response
                }
            )

        return {
            "message": "Trade evaluation placeholder",
            **response
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)})
