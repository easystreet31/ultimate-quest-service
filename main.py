from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

# Input schema for a single player row
class PlayerRow(BaseModel):
    Player: str
    SP: int
    QP: int
    Rank: int

# Input schema for the full payload
class TradePayload(BaseModel):
    easystreet31: List[PlayerRow]
    finkleiseinhorn: List[PlayerRow]
    dustercrusher: List[PlayerRow]
    leaderboard: List[Dict[str, Any]]
    the_collection: List[Dict[str, Any]]
    trade_file: List[Dict[str, Any]]

@app.post("/evaluate_trade_json")
def evaluate_trade_json(payload: TradePayload):
    try:
        # Convert Pydantic models to dict
        data = payload.dict()

        # TODO: Replace with your actual trade evaluation logic
        # For now, just return a dummy JSON structure
        result = {
            "session_state": {
                "holdings": [
                    "holdings_easystreet31.json",
                    "holdings_FinkleIsEinhorn.json",
                    "holdings_DusterCrusher.json"
                ],
                "leaderboard": "leaderboard.json",
                "the_collection": "The Collection.json",
                "trade": "trade.json"
            },
            "per_account": [
                {
                    "account": "Easystreet31",
                    "player_lines": [
                        {
                            "player": "Claude Giroux",
                            "before_sp": 9,
                            "after_sp": 7,
                            "delta_sp": -2,
                            "before_rank": 5,
                            "after_rank": 99,
                            "before_qp": 0,
                            "after_qp": 0,
                            "delta_qp": 0
                        }
                    ],
                    "net_delta_sp": 0,
                    "net_delta_qp": 0,
                    "fragile": []
                }
            ],
            "overall_qp_summary": 0
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
