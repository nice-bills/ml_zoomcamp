from typing import Dict, Any
import pickle
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="lead-scoring-prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(trader):
    result = pipeline.predict_proba(trader)[0, 1]
    return float(result)

@app.post("/predict")
def predict(trader: Dict[str, Any]):
    prob = predict_single(trader)

    return {
        "trader_prob": prob,
        "type": bool(prob >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)