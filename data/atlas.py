import os
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from dotenv import load_dotenv
from typing import Optional
import numpy as np

load_dotenv()

# --- Connection ---

def get_db():
    """
    Returns a connected MongoDB database instance.
    Call this once at the top of any script that needs Atlas access.
    """
    client = MongoClient(os.getenv("MONGODB_URI"))
    return client[os.getenv("MONGODB_DB")]


# --- Customers ---

def insert_customers(customers: list[dict], db) -> None:
    """
    Bulk insert a list of customer dicts into the customers collection.
    Used during data generation. Skips duplicates by customer_id.
    """
    if not customers:
        return
    operations = [
        UpdateOne(
            {"customer_id": c["customer_id"]},
            {"$setOnInsert": c},
            upsert=True
        )
        for c in customers
    ]
    result = db.customers.bulk_write(operations, ordered=False)
    print(f"  inserted: {result.upserted_count} | skipped (existing): {result.matched_count}")


def get_customer(customer_id: str, db) -> Optional[dict]:
    """Fetch a single customer by customer_id."""
    return db.customers.find_one({"customer_id": customer_id}, {"_id": 0})


def get_customers(limit: int = 100, db=None) -> list[dict]:
    """Fetch a batch of customers. Used to seed the training pool."""
    return list(db.customers.find({}, {"_id": 0}).limit(limit))


def get_all_customers(db) -> list[dict]:
    """
    Fetch all customers into memory.
    Called once at training start — do not call per-step.
    Free tier Atlas has rate limits; bulk load once, iterate in memory.
    """
    return list(db.customers.find({}, {"_id": 0}))


def update_churn_label(customer_id: str, churned: bool, db) -> None:
    """Record the final churn outcome for a customer after an episode."""
    db.customers.update_one(
        {"customer_id": customer_id},
        {"$set": {"churn_label": churned}}
    )


# --- Interventions ---

def insert_intervention(intervention: dict, db) -> None:
    """Insert a single intervention event."""
    db.interventions.insert_one({k: v for k, v in intervention.items() if k != "_id"})


def insert_interventions_bulk(interventions: list[dict], db) -> None:
    """
    Bulk insert intervention events.
    Used at end of evaluation run to write all outcomes at once.
    """
    if not interventions:
        return
    db.interventions.insert_many(
        [{k: v for k, v in i.items() if k != "_id"} for i in interventions]
    )


def get_interventions_for_customer(customer_id: str, db) -> list[dict]:
    """Fetch all interventions for a given customer."""
    return list(db.interventions.find({"customer_id": customer_id}, {"_id": 0}))


# --- Training Runs ---

def insert_training_run(run: dict, db) -> None:
    """Insert a training run summary document."""
    db.training_runs.insert_one({k: v for k, v in run.items() if k != "_id"})


def get_training_run(run_id: str, db) -> Optional[dict]:
    """Fetch a single training run by run_id."""
    return db.training_runs.find_one({"run_id": run_id}, {"_id": 0})


def get_training_runs(limit: int = 20, db=None) -> list[dict]:
    """Fetch recent training runs, most recent first."""
    return list(
        db.training_runs.find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )


# --- Vector Search ---

def find_similar_customers(feature_vector: list[float], db, n: int = 5) -> list[dict]:
    """
    Run an Atlas Vector Search query to find the n most similar customers
    by behavioral feature vector.

    Joins intervention history so callers can see what worked on similar customers.

    NOTE: Requires a Vector Search index named 'customer_vector_index' on the
    customers collection. Create this manually in the Atlas UI before calling.
    See AGENT.md for index configuration.

    Args:
        feature_vector: 10-dimensional normalized float list
        db: connected database instance
        n: number of similar customers to return

    Returns:
        List of customer dicts with embedded intervention_history array
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": "customer_vector_index",
                "path": "feature_vector",
                "queryVector": feature_vector,
                "numCandidates": 50,
                "limit": n
            }
        },
        {
            "$lookup": {
                "from": "interventions",
                "localField": "customer_id",
                "foreignField": "customer_id",
                "as": "intervention_history"
            }
        },
        {
            "$project": {
                "_id": 0,
                "customer_id": 1,
                "plan_type": 1,
                "churn_probability": 1,
                "churn_label": 1,
                "tenure_months": 1,
                "feature_vector": 1,
                "intervention_history": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    return list(db.customers.aggregate(pipeline))


def summarize_similar_interventions(similar_customers: list[dict]) -> dict:
    """
    Given a list of similar customers from find_similar_customers(),
    summarize what interventions were taken and what outcomes resulted.

    Returns a dict of action_index -> retention_rate for use in evaluation.
    """
    action_outcomes = {i: {"retained": 0, "churned": 0} for i in range(5)}

    for customer in similar_customers:
        for intervention in customer.get("intervention_history", []):
            action = intervention.get("agent_action_index")
            outcome = intervention.get("outcome")
            if action is not None and outcome in ("retained", "churned"):
                action_outcomes[action][outcome] += 1

    summary = {}
    for action, counts in action_outcomes.items():
        total = counts["retained"] + counts["churned"]
        summary[action] = counts["retained"] / total if total > 0 else None

    return summary


if __name__ == "__main__":
    # Smoke test — verifies connection and collection access
    db = get_db()
    print("Collections:", db.list_collection_names())
    print("Customers count:", db.customers.count_documents({}))
    print("Interventions count:", db.interventions.count_documents({}))
    print("Training runs count:", db.training_runs.count_documents({}))