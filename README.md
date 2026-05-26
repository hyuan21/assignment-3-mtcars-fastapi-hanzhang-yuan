# MTCARS FastAPI

A reproducible end-to-end machine-learning service that predicts a vehicle's
fuel economy (`mpg`, miles per gallon) from its weight and horsepower. The
model is a linear regression trained on the classic `mtcars` dataset and is
served by a FastAPI application that runs locally in Podman and is deployed
to Google Cloud Run.

> **Assignment 3** — MTCARS FastAPI Deployment

## Table of contents

1. [Project overview](#1-project-overview)
2. [Model description](#2-model-description)
3. [Repository structure](#3-repository-structure)
4. [Local setup](#4-local-setup)
5. [Train the model](#5-train-the-model)
6. [Run the API locally (no container)](#6-run-the-api-locally-no-container)
7. [Run with Podman](#7-run-with-podman)
8. [API documentation](#8-api-documentation)
9. [Example `curl` calls](#9-example-curl-calls)
10. [Tests](#10-tests)
11. [Deploy to Google Cloud Run](#11-deploy-to-google-cloud-run)
12. [Push the project to GitHub](#12-push-the-project-to-github)
13. [Production-oriented features](#13-production-oriented-features)

---

## 1. Project overview

This repo trains a `LinearRegression` from `scikit-learn` on `mtcars.csv`,
saves the fitted model to disk with `joblib`, and serves predictions through
a FastAPI application. The application is containerised with a Dockerfile,
runs locally under Podman, and is deployed to Cloud Run as the public API.

The goal is to demonstrate the full lifecycle of a small ML service:
**EDA → train → serve → containerise → deploy → test**.

## 2. Model description

* **Response variable:** `mpg` (miles per gallon).
* **Predictor variables:** `wt` (vehicle weight in 1000 lbs) and
  `hp` (gross horsepower).
* **Algorithm:** ordinary least squares linear regression
  (`sklearn.linear_model.LinearRegression`).

### Why `wt` and `hp`?

The four variables most correlated with `mpg` on this 32-row dataset are
`wt` (r = -0.87), `cyl` (-0.85), `disp` (-0.85) and `hp` (-0.78), but they
are heavily collinear with each other. A Variance Inflation Factor (VIF)
analysis showed `disp` at 10.37 and `cyl` at 6.74 when all four were used
together, while a `wt + hp` model had VIFs of only 1.77 each — well below
the common 5 threshold.

`wt` and `hp` together capture two intuitively distinct effects — vehicle
mass and engine power — without redundancy. The resulting model performance:

| metric        | value  |
| ------------- | ------ |
| R² (in-sample) | 0.827  |
| RMSE          | 2.469  |
| MAE           | 1.901  |
| R² (5-fold CV) | 0.740  |

Fitted equation:

```
mpg ≈ 37.227  − 3.878 · wt  − 0.032 · hp
```

Re-run `python scripts/train_model.py` to regenerate the model artifact
from scratch.

## 3. Repository structure

```
.
├── README.md                  # this file
├── mtcars.csv                 # dataset (32 rows × 12 columns)
├── Dockerfile                 # container image definition
├── .dockerignore              # files excluded from the build context
├── .gitignore
├── requirements.txt           # pinned Python dependencies
├── app/
│   └── main.py                # FastAPI app: /health, /ready, /predict
├── scripts/
│   └── train_model.py         # trains and saves models/model.pkl
├── models/
│   └── model.pkl              # joblib artifact loaded by the API
└── tests/
    └── test_api.py            # pytest suite (10 tests)
```

## 4. Local setup

### Prerequisites

* Python 3.11+ (works on 3.10 as well)
* [Podman](https://podman.io/) for the container workflow
* Optional: [`uv`](https://docs.astral.sh/uv/) for faster envs

### Create a virtual environment

```bash
# with uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# or with the stock tooling
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Train the model

```bash
python scripts/train_model.py
```

This writes `models/model.pkl`. The FastAPI app loads it at startup, so
this step must happen **before** building the container.

## 6. Run the API locally (no container)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
# or
python -m app.main
```

Open <http://localhost:8080/docs> to see the auto-generated Swagger UI.

## 7. Run with Podman

```bash
# 1. Make sure the model artifact exists (see step 5).
python scripts/train_model.py

# 2. Build the image.
podman build -t mtcars-fastapi .

# 3. Run the container, exposing port 8080.
podman run --rm -p 8080:8080 mtcars-fastapi

# 4. Health-check it from another terminal.
curl http://localhost:8080/health
```

To override the model path or log level at runtime:

```bash
podman run --rm -p 8080:8080 \
    -e LOG_LEVEL=DEBUG \
    -e MODEL_PATH=/app/models/model.pkl \
    mtcars-fastapi
```

## 8. API documentation

Interactive docs are generated automatically by FastAPI and available at
`/docs` (Swagger UI) and `/redoc` (ReDoc).

| Method | Path       | Purpose                                                   |
| ------ | ---------- | --------------------------------------------------------- |
| GET    | `/health`  | Liveness check; returns `{"status": "ok"}`.               |
| GET    | `/ready`   | Readiness check; 200 when the model is loaded, 503 otherwise. |
| POST   | `/predict` | Returns predicted `mpg` for the given `wt` and `hp`.      |

### `POST /predict` request schema

```json
{
  "wt": 2.62,
  "hp": 110
}
```

* `wt`: float, must be `> 0`. Vehicle weight in 1000 lbs.
* `hp`: float, must be `> 0`. Gross horsepower.

### `POST /predict` response schema

```json
{
  "predicted_mpg": 22.5878,
  "features_used": ["wt", "hp"],
  "model_intercept": 37.2273,
  "model_coefficients": {
    "wt": -3.8778,
    "hp": -0.0318
  }
}
```

### Error responses

| HTTP status | Cause                                                         |
| ----------- | ------------------------------------------------------------- |
| 422         | Pydantic validation failure (missing field, wrong type, etc.) |
| 500         | Unexpected exception during prediction                        |
| 503         | Model artifact missing or failed to load                      |

## 9. Example `curl` calls

### Health check

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

### Readiness check

```bash
curl http://localhost:8080/ready
# {"status":"ready","model_loaded":true,...}
```

### Predict

```bash
curl -X POST "http://localhost:8080/predict" \
    -H "Content-Type: application/json" \
    -d '{"wt": 2.62, "hp": 110}'
# {"predicted_mpg":22.5878,"features_used":["wt","hp"],...}
```

### Validation error (missing field)

```bash
curl -X POST "http://localhost:8080/predict" \
    -H "Content-Type: application/json" \
    -d '{"wt": 2.62}'
# 422 with details about the missing "hp" field
```

## 10. Tests

The repo ships with a pytest suite that exercises every endpoint, including
input validation and the "model missing" failure mode.

```bash
pytest -v
```

Expected: **10 passed**. Coverage:

* `/health` returns 200.
* `/ready` returns 200 when loaded and 503 when not.
* `/predict` returns a plausible mpg for a known input.
* heavier car → lower predicted mpg (sanity property).
* `/predict` rejects missing fields, wrong types, and non-positive weight.
* `/predict` returns 503 when the model is unavailable.

## 11. Deploy to Google Cloud Run

Cloud Run runs container images on demand and exposes them on a public HTTPS
URL. The image registry used below is Google Artifact Registry.

> **One-time setup:** make sure you have `gcloud` installed, are logged in
> (`gcloud auth login`), have selected a billing-enabled project, and have
> enabled the Cloud Run + Artifact Registry APIs.

### Step 1 — set environment variables

```bash
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1
export REPO=mtcars
export IMAGE=mtcars-fastapi
export TAG=v1
```

### Step 2 — enable required APIs

```bash
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    --project "$PROJECT_ID"
```

### Step 3 — create an Artifact Registry repo (once per project)

```bash
gcloud artifacts repositories create "$REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --project "$PROJECT_ID"
```

### Step 4 — authenticate Podman to the registry

```bash
gcloud auth print-access-token | \
    podman login -u oauth2accesstoken --password-stdin \
    "$REGION-docker.pkg.dev"
```

### Step 5 — build and tag the image

```bash
podman build -t "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG" .
```

### Step 6 — push the image

```bash
podman push "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG"
```

### Step 7 — deploy to Cloud Run

```bash
gcloud run deploy mtcars-fastapi \
    --image "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --project "$PROJECT_ID"
```

`gcloud` will print the service URL. Smoke-test it:

```bash
curl https://<your-service>-<hash>-uc.a.run.app/health
curl -X POST "https://<your-service>-<hash>-uc.a.run.app/predict" \
    -H "Content-Type: application/json" \
    -d '{"wt": 2.62, "hp": 110}'
```

### Deployed URL

> **TODO: replace this line with your Cloud Run URL once deployed**, e.g.
> `https://mtcars-fastapi-xxxxxxxxxx-uc.a.run.app`.

## 12. Push the project to GitHub

```bash
# from this directory
git init
git add .
git commit -m "Initial commit: MTCARS FastAPI assignment 3"

# Create a new empty repo on github.com (e.g. mtcars-fastapi-api) first,
# then:
git branch -M main
git remote add origin https://github.com/<your-username>/mtcars-fastapi-api.git
git push -u origin main
```

Make sure the `models/model.pkl` artifact is committed — the assignment
requires the repo to be self-contained. (It is small, ~1 KB.)

## 13. Production-oriented features

This service intentionally exercises several "production-minded" practices:

* **Liveness vs readiness** — `/health` and `/ready` are separate endpoints
  (the readiness probe actually checks model availability and returns 503
  when unavailable, which is what an orchestrator needs).
* **Pydantic validation** — types and positivity constraints are enforced;
  invalid input returns 422 with a helpful message.
* **Graceful model-load failure** — the app starts even if the model file
  is missing; `/ready` and `/predict` both report the problem clearly.
* **Structured logging** — every prediction is logged with its inputs and
  output; log level is controlled by the `LOG_LEVEL` env var.
* **Environment-driven config** — `MODEL_PATH`, `LOG_LEVEL`, and `PORT` can
  all be overridden without rebuilding the image.
* **Non-root container** — the image runs as `appuser` (UID 1000).
* **Layered Dockerfile** — dependencies are installed before source code is
  copied in, so source-only edits don't bust the dependency cache.
* **Auto-generated OpenAPI docs** — `/docs` and `/redoc` are always available.
* **CI-friendly tests** — `pytest` covers happy paths, validation errors,
  and the missing-model failure mode.
