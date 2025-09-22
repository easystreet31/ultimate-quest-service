from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/evaluate_trade_json")
async def evaluate_trade_json(request: Request):
    try:
        data = await request.json()

        required_keys = [
            "easystreet31",
            "finkleiseinhorn",
            "dustercrusher",
            "leaderboard",
            "the_collection",
            "trade_file"
        ]

        missing = [k for k in required_keys if k not in data]

        # 🔍 Debug: Echo back what was actually received
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

        # ✅ Placeholder for real evaluation logic
        return {
            "message": "Trade evaluation placeholder",
            **response
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
