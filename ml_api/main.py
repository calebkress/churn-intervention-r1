"""
fastapi/main.py
===============
ENDPOINTS:
  GET  /health                  — liveness check
  POST /infer                   — run trained policy on a customer observation
  GET  /insights/{customer_id}  — LangChain recommendation (calls insights.py)
  GET  /similar/{customer_id}   — raw Vector Search results (no LLM)
  POST /train                   — trigger a new training run (background task)
  GET  /train/status            — check background training status

RUN:
  From project root:
    uvicorn fastapi.main:app --reload --port 8000

  Or with PYTHONPATH set (required for sibling imports):
    PYTHONPATH=. uvicorn fastapi.main:app --reload --port 8000
"""

import os
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from stable_baselines3 import PPO

load_dotenv()

# ---------------------------------------------------------------------------
# Sibling package imports — works when run with PYTHONPATH=. from project root
# ---------------------------------------------------------------------------
from data.atlas import get_db, get_customer, find_similar_customers
from lc.insights import get_customer_insight

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = "models/ppo_churn_b4193c90"   # canonical model — no .zip suffix (SB3 convention)

ACTION_NAMES = {
    0: "do_nothing",
    1: "send_email_offer",
    2: "outbound_call",
    3: "discount_10pct",
    4: "escalate_to_retention",
}

# ---------------------------------------------------------------------------
# App state — loaded once at startup, shared across all requests
# ---------------------------------------------------------------------------

class AppState:
    model: Optional[PPO] = None
    db = None
    mongo_client = None
    training_status: dict = {"running": False, "last_run_id": None, "error": None}

state = AppState()

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup and once at shutdown.
    Load the model and open the Atlas connection here so every
    request reuses them instead of paying the connection cost each time.
    """
    print("Starting up FastAPI...")

    # Load PPO model
    try:
        state.model = PPO.load(MODEL_PATH)
        print(f"  Model loaded: {MODEL_PATH}.zip")
    except FileNotFoundError:
        print(f"  WARNING: Model not found at {MODEL_PATH}.zip — /infer will fail")
        state.model = None

    # Open Atlas connection
    try:
        state.db = get_db()
        state.mongo_client = state.db.client
        print(f"  Atlas connected: {state.db.name}")
    except Exception as e:
        print(f"  WARNING: Atlas connection failed — {e}")
        state.db = None

    yield  # app runs here

    # Shutdown
    if state.mongo_client:
        state.mongo_client.close()
    print("FastAPI shut down.")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Churn Intervention RL",
    description="ML inference and LangChain insights for the churn intervention system.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3001"],  # Vite + Express
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    """
    The 14-dimensional observation vector from churn_env.py.
    Matches Box(float32, shape=(14,)) exactly — see AGENT.md for field order.
    """
    observation: list[float]

class InferResponse(BaseModel):
    action_index: int
    action_name: str
    observation: list[float]

class InsightResponse(BaseModel):
    customer_id: str
    insight: str
    similar_customers: list[dict]

class TrainResponse(BaseModel):
    message: str
    started_at: str

class TrainStatusResponse(BaseModel):
    running: bool
    last_run_id: Optional[str]
    error: Optional[str]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_db():
    """Raise 503 if Atlas isn't connected."""
    if state.db is None:
        raise HTTPException(status_code=503, detail="Atlas connection unavailable")

def _require_model():
    """Raise 503 if the PPO model isn't loaded."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check model path")

def _get_customer_or_404(customer_id: str) -> dict:
    """Fetch a customer from Atlas or raise 404."""
    customer = get_customer(customer_id, state.db)
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return customer

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Liveness check. Express hits this to confirm FastAPI is up before proxying.
    Returns model and db status so you can diagnose startup issues quickly.
    """
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "db_connected": state.db is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/infer", response_model=InferResponse)
def infer(request: InferRequest):
    """
    Run the trained PPO policy on a single observation vector.

    The caller (Express) is responsible for building the 14-dim observation
    from a customer document. The order must match churn_env.py exactly.

    Returns the action index and its human-readable name.

    Example request body:
        { "observation": [0.6, 0.5, 0.28, 0.25, 0.03, 0.2, 0.0, 0.6, 0.25, 0.67, 0.0, 0.0, 1.0, 1.0] }
    """
    _require_model()

    if len(request.observation) != 14:
        raise HTTPException(
            status_code=422,
            detail=f"Observation must be 14-dimensional. Got {len(request.observation)}."
        )

    obs = np.array(request.observation, dtype=np.float32)
    action, _ = state.model.predict(obs, deterministic=True)
    action_idx = int(action)

    return InferResponse(
        action_index=action_idx,
        action_name=ACTION_NAMES[action_idx],
        observation=request.observation,
    )


@app.get("/insights/{customer_id}", response_model=InsightResponse)
def insights(customer_id: str):
    """
    Generate a plain-English intervention recommendation for a customer.

    Flow:
      1. Fetch the customer document from Atlas
      2. Pass feature_vector to LangChain insights chain
      3. Chain runs Atlas Vector Search → formats context → calls gpt-4o-mini
      4. Returns 2-3 sentence recommendation + similar customer metadata

    This is the demo endpoint — it shows Atlas Vector Search + LangChain
    working together with a single Atlas connection.
    """
    _require_db()

    customer = _get_customer_or_404(customer_id)

    if not customer.get("feature_vector"):
        raise HTTPException(
            status_code=422,
            detail=f"Customer {customer_id} has no feature_vector. Re-run data/generate.py."
        )

    try:
        result = get_customer_insight(customer, client=state.mongo_client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")

    return InsightResponse(**result)


@app.get("/similar/{customer_id}")
def similar(customer_id: str, n: int = 5):
    """
    Return the N most similar customers via Atlas Vector Search.
    No LLM call — raw retrieval results only.

    Useful for the React dashboard to show who the agent is comparing against
    without paying for a gpt-4o-mini call.

    Query param:
        n: number of similar customers to return (default 5, max 20)
    """
    _require_db()

    if n > 20:
        raise HTTPException(status_code=422, detail="n must be 20 or less")

    customer = _get_customer_or_404(customer_id)

    feature_vector = customer.get("feature_vector")
    if not feature_vector:
        raise HTTPException(
            status_code=422,
            detail=f"Customer {customer_id} has no feature_vector."
        )

    similar_customers = find_similar_customers(feature_vector, state.db, n=n)

    return {
        "customer_id": customer_id,
        "n": len(similar_customers),
        "similar_customers": similar_customers,
    }


@app.post("/train", response_model=TrainResponse)
def train():
    """
    Trigger a new PPO training run in a background thread.

    The spec says not to retrain the canonical model — this endpoint exists
    so the demo can show the system is capable of retraining, but in practice
    you'd only call it if you explicitly want a new model.

    Returns immediately. Poll /train/status to check progress.
    """
    if state.training_status["running"]:
        raise HTTPException(status_code=409, detail="Training is already running")

    def _run_training():
        state.training_status["running"] = True
        state.training_status["error"] = None
        try:
            from agent.train import train as run_train
            summary = run_train()
            state.training_status["last_run_id"] = summary.get("run_id")
        except Exception as e:
            state.training_status["error"] = str(e)
        finally:
            state.training_status["running"] = False

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()

    return TrainResponse(
        message="Training started in background. Poll /train/status for updates.",
        started_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/train/status", response_model=TrainStatusResponse)
def train_status():
    """
    Check whether a background training run is in progress.
    """
    return TrainStatusResponse(**state.training_status)