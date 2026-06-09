import os
import sys
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pymongo import MongoClient

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Human-readable names for the 5 action indices — matches env/customer.py
ACTION_NAMES = {
    0: "do_nothing",
    1: "send_email_offer",
    2: "outbound_call",
    3: "discount_10pct",
    4: "escalate_to_retention",
}

# Number of similar customers to retrieve from Atlas Vector Search
N_SIMILAR = 5

# ---------------------------------------------------------------------------
# Atlas + LangChain setup
# ---------------------------------------------------------------------------


def _get_mongo_client() -> MongoClient:
    """Return a connected MongoClient using MONGODB_URI from .env."""
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise EnvironmentError("MONGODB_URI is not set in your .env file.")
    return MongoClient(uri)


def _build_llm() -> ChatOpenAI:
    """Return gpt-4o-mini — cheap, fast, sufficient for 2-3 sentence summaries."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,          # low temp = consistent, professional tone
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def _format_intervention_history(intervention_history: list[dict]) -> str:
    """
    Convert a customer's raw intervention records into a readable summary string.

    Example output:
      - outbound_call → retained
      - discount_10pct → churned
      - send_email_offer → retained

    If no interventions exist, returns a note saying so.
    """
    if not intervention_history:
        return "  (no interventions recorded)"

    lines = []
    for record in intervention_history:
        record.pop('_id', None)
        action_idx = record.get("agent_action_index")
        outcome = record.get("outcome", "unknown")
        action_name = ACTION_NAMES.get(action_idx, f"action_{action_idx}")
        lines.append(f"  - {action_name} → {outcome}")
    return "\n".join(lines)


def _format_similar_customers_context(similar_customers: list[dict]) -> str:
    """
    Take the list of customer dicts returned by find_similar_customers()
    and build the full context block we'll inject into the LLM prompt.
    """
    if not similar_customers:
        return "No similar customers found in the database."

    blocks = []
    for i, customer in enumerate(similar_customers, start=1):
        plan = customer.get("plan_type", "unknown")
        churn_prob = customer.get("churn_probability", 0.0)
        tenure = customer.get("tenure_months", 0)
        score = customer.get("score", 0.0)
        history = customer.get("intervention_history", [])

        history_text = _format_intervention_history(history)

        blocks.append(
            f"Customer {i} (similarity score: {score:.3f}):\n"
            f"  Plan: {plan} | Tenure: {tenure} months | "
            f"Churn probability: {churn_prob:.0%}\n"
            f"  Intervention history:\n{history_text}"
        )

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a telecom customer retention analyst. \
Given data about similar customers and what interventions worked on them, \
provide a concise recommendation for how to approach the current customer. \
Be specific about which intervention type showed the best retention rate and why. \
Keep your response to 2-3 sentences."""

HUMAN_PROMPT = """Current customer profile:
  Plan: {plan_type}
  Tenure: {tenure_months} months
  Churn probability: {churn_probability:.0%}

Similar customers and their outcomes:
{similar_context}

What intervention would you recommend for this customer?"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_customer_insight(
    customer: dict,
    client: Optional[MongoClient] = None,
) -> dict:
    """
    Generate a plain-English intervention recommendation for a customer.

    This is the function FastAPI calls at GET /insights/{customer_id}.

    Args:
        customer: dict with at minimum:
                    - feature_vector: list[float] (10 dims)
                    - plan_type: str
                    - tenure_months: int
                    - churn_probability: float
                    - customer_id: str
        client:   optional MongoClient (pass one in to reuse connections;
                  if None, a new client is created from .env)

    Returns:
        {
            "customer_id": str,
            "insight": str,           # 2-3 sentence LLM recommendation
            "similar_customers": [...] # raw similar customer metadata for UI
        }

    Raises:
        EnvironmentError: if MONGODB_URI or OPENAI_API_KEY are missing
        ValueError: if customer has no feature_vector
    """
    feature_vector = customer.get("feature_vector")
    if not feature_vector or len(feature_vector) != 10:
        raise ValueError(
            f"customer must have a 10-dimensional feature_vector. "
            f"Got: {feature_vector}"
        )

    # Create client if not provided — FastAPI will pass a persistent one
    owns_client = client is None
    if owns_client:
        client = _get_mongo_client()

    try:
        from data.atlas import find_similar_customers
        llm = _build_llm()

        # --- Step 1: Retrieve similar customers via Atlas Vector Search ---
        # Uses find_similar_customers() from data/atlas.py which runs the
        # $vectorSearch aggregation + $lookup for intervention history directly.
        similar_customers = find_similar_customers(feature_vector, client[os.environ.get("MONGODB_DB", "churn_rl")], n=N_SIMILAR)

        # --- Step 2: Format context for the LLM ---
        similar_context = _format_similar_customers_context(similar_customers)

        # --- Step 3: Build the prompt inputs ---
        prompt_inputs = {
            "plan_type": customer.get("plan_type", "unknown"),
            "tenure_months": customer.get("tenure_months", 0),
            "churn_probability": customer.get("churn_probability", 0.0),
            "similar_context": similar_context,
        }

        # --- Step 4: Invoke LLM chain ---
        chain = PROMPT | llm | StrOutputParser()
        insight_text = chain.invoke(prompt_inputs)

        # --- Step 5: Return structured result ---
        return {
            "customer_id": customer.get("customer_id"),
            "insight": insight_text.strip(),
            "similar_customers": [
                {
                    "customer_id": c.get("customer_id"),
                    "plan_type": c.get("plan_type"),
                    "tenure_months": c.get("tenure_months"),
                    "churn_probability": c.get("churn_probability"),
                    "similarity_score": c.get("score"),
                    "intervention_history": c.get("intervention_history", []),
                }
                for c in similar_customers
            ],
        }

    finally:
        # Only close if we opened it — FastAPI passes a persistent client
        if owns_client:
            client.close()


# ---------------------------------------------------------------------------
# Smoke test — run directly: python -m langchain.insights
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick smoke test. Pulls a real customer from Atlas, runs the full chain,
    and prints the recommendation.

    Usage:
        python -m langchain.insights

    Requires:
        - .env with MONGODB_URI and OPENAI_API_KEY
        - customer_vector_index created in Atlas UI
        - At least one customer document in the customers collection
    """
    from data.atlas import get_db

    print("Connecting to Atlas...")
    db = get_db()
    mongo_client = db.client

    # Grab one customer with a feature_vector
    sample = db.customers.find_one(
        {"feature_vector": {"$exists": True}},
        {"_id": 0}
    )

    if not sample:
        print("ERROR: No customers found in Atlas. Run data/generate.py first.")
        sys.exit(1)

    print(f"Testing with customer: {sample['customer_id']}")
    print(f"  Plan: {sample['plan_type']} | "
          f"Tenure: {sample['tenure_months']}mo | "
          f"Churn prob: {sample['churn_probability']:.0%}")
    print()

    result = get_customer_insight(sample, client=mongo_client)

    print("=" * 60)
    print("INSIGHT:")
    print(result["insight"])
    print()
    print(f"Based on {len(result['similar_customers'])} similar customers.")
    print("=" * 60)