from typing import Dict, Any
import pickle
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="lead-scoring-prediction")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    prob = predict_single(customer)

    return {
        "lead_score_probability": prob,
        "converted": bool(prob >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)