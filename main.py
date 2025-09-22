from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

class TradePayload(BaseModel):
    easystreet31: List[Any]
    finkleiseinhorn: List[Any]
    dustercrusher: List[Any]
    leaderboard: List[Any]
    the_collection: List[Any]
    trade: Dict[str, Any]

@app.post("/evaluate_trade_json")
async def evaluate_trade_json(payload: TradePayload):
    data = payload.dict()

    # Validate required keys (redundant if using Pydantic, but kept for clarity)
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
        raise HTTPException(status_code=400, detail={"error": "Missing required keys", **response})

    # Placeholder evaluation logic
    return {
        "message": "Trade evaluation placeholder",
        **response
    }
