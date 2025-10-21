# Machine Learning (ML) Deployment — module_5 (FastAPI + uvicorn)

This module shows a compact way to serve a trained classifier as an HTTP service using FastAPI and uvicorn. It includes:
- FastAPI app: `module_5/predict.py` (endpoints: `predict`, `predict_single`)
- Docker recipe: `module_5/Dockerfile`
- Place your model artifacts in this folder (examples: `model1.bin`)

Note: this project uses "uv" as the local environment initializer (run `uv init` below). After that we run the FastAPI app with uvicorn.

Quick checklist before starting
- Put serialized artifacts (DictVectorizer, model) in module_5 (e.g. `model1.bin`).
- Confirm `module_5/predict.py` references the correct filenames or set env vars DV_FILE / MODEL_FILE.

1) Initialize local environment (uv)
Run this once in the module directory to initialize the development environment:
```bash
cd module_5
uv init
```
If `uv` is not present in your environment, use your preferred tool to prepare a Python runtime (venv/pip, poetry, etc.). The purpose of `uv init` here is to show the project was initialized with the "uv" flow.

2) Run the API locally (uvicorn)
Development (auto-reload)
```bash
cd module_5
uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
```
Production-like
```bash
cd module_5
uvicorn predict:app --host 0.0.0.0 --port 9696 
```

3) Model artifacts and usage
Place your binary artifact in this folder (defaults used by the app):
- model1.bin — trained classifier (pickle/joblib)

Example from Python (load and infer)
```python
from typing import Dict, Any
import pickle
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="lead-scoring-prediction")

with open('pipeline_v2.bin', 'rb') as f_in:
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
```

4) API endpoints (implemented in module_5/predict.py)
- POST /predict  
  Accepts a JSON array of feature dicts. Returns per-record prediction and probability (if available).
- POST /predict_single  
  Accepts a single JSON object and returns one prediction + probability.

Example requests

Python client
```python 
import requests

url = "http://localhost:9696/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json= client)
predictions = response.json()

print(predictions)
```

Single (curl)
```bash
curl -sX POST "http://localhost:8000/predict_single" \
  -H "Content-Type: application/json" \
  -d '"lead_source": "organic_search", "number_of_courses_viewed": 4, "annual_income": 80304.0'
```

5) Docker: build & run
The included Dockerfile packages the app and model files. Place dv/model bin files in module_5 or update the Dockerfile to COPY them.

Build image (from repo root)
```bash
docker build -t module_5_api:latest module_5
```

Run container and map port 9696
```bash
docker run -d --name module_5_api -p 9696:9696 module_5_api:latest
```

Inspect and manage
```bash
docker ps
docker logs -f module_5_api
docker stop module_5_api && docker rm module_5_api
```

Update the container after changes
```bash
docker build -t module_5_api:latest module_5
docker stop module_5_api && docker rm module_5_api
docker run -d --name module_5_api -p 9696:9696 module_5_api:latest
```

6) Configuration and environment variables
The app supports overriding filenames via environment variables:
- DV_FILE — path to vectorizer (default: dv.bin)
- MODEL_FILE — path to model binary (default: model1.bin)

Example:
```bash
export DV_FILE=my_dv.bin
export MODEL_FILE=my_model.bin
uvicorn predict:app --host 0.0.0.0 --port 8000
```

7) Production considerations (short)
- Use input validation (pydantic) and add health/metrics endpoints.
- Run multiple workers or place behind an orchestrator for scaling.
- Protect endpoints (auth, rate limits) before exposing publicly.
- Log and monitor prediction distributions and latencies; avoid logging sensitive inputs.

Files of interest
- module_5/predict.py — FastAPI app (endpoints: predict, predict_single)  
- module_5/predict_2.py — alternate/experimental implementation  
- module_5/Dockerfile — container recipe  
- module_5/pyproject.toml — dependency hints
