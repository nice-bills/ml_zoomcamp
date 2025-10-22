import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

class Trader(BaseModel):
    active_weeks: int
    total_volume: float
    trader_activity_status: str
    trader_volume_status: str
    trader_weekly_frequency_status: str
    tx_count_365d: int
    wallet: str

with open("logreg_pipeline.pkl", "rb") as f_in:
    pipeline = joblib.load(f_in)

app = FastAPI()

def predict_single(trader_dict):
    df = pd.DataFrame([trader_dict])

    expected_cols = pipeline.feature_names_in_
    missing_cols = set(expected_cols) - set(df.columns)
    for col in missing_cols:
        df[col] = None  

    df = df[expected_cols]

    result = pipeline.predict_proba(df)[0, 1]
    return result


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.post("/predict")
def predict(trader: Trader):
    trader_dict = trader.dict()
    prob = predict_single(trader_dict)
    pred = prob >= 0.5
    return {"probability": float(prob), "prediction": bool(pred)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
