import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm
from data.atlas import get_db, insert_customers
from env.customer import generate_customer

# --- Configuration ---

NUM_CUSTOMERS = 10_000
BATCH_SIZE = 500
RANDOM_SEED = 42


def generate_customers(n: int) -> list[dict]:
    """
    Generate n synthetic customers and return as a list of dicts
    ready for Atlas insertion.
    """
    customers = []
    for _ in tqdm(range(n), desc="Generating customers"):
        customer = generate_customer()
        customers.append(customer.to_dict())
    return customers


def seed_atlas(customers: list[dict], db) -> None:
    """
    Insert customers into Atlas in batches.
    Batching avoids hitting document size limits on bulk writes
    and gives progress visibility on large inserts.
    """
    total = len(customers)
    print(f"\nSeeding {total} customers to Atlas in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch = customers[i: i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches}", end=" ")
        insert_customers(batch, db)

    print("\nDone.")


def print_summary(customers: list[dict]) -> None:
    """
    Print a summary of the generated population so you can sanity check
    the distributions look realistic before committing to Atlas.
    """
    churn_probs = [c["churn_probability"] for c in customers]
    plan_counts = {"prepaid": 0, "postpaid": 0, "enterprise": 0}
    for c in customers:
        plan_counts[c["plan_type"]] += 1

    high_risk = sum(1 for p in churn_probs if p > 0.6)
    med_risk = sum(1 for p in churn_probs if 0.3 <= p <= 0.6)
    low_risk = sum(1 for p in churn_probs if p < 0.3)

    print("\n--- Population Summary ---")
    print(f"Total customers:     {len(customers)}")
    print(f"\nPlan distribution:")
    for plan, count in plan_counts.items():
        pct = count / len(customers) * 100
        print(f"  {plan:<12} {count:>5}  ({pct:.1f}%)")
    print(f"\nChurn risk distribution:")
    print(f"  High  (>0.6):   {high_risk:>5}  ({high_risk / len(customers) * 100:.1f}%)")
    print(f"  Medium (0.3-0.6): {med_risk:>5}  ({med_risk / len(customers) * 100:.1f}%)")
    print(f"  Low   (<0.3):   {low_risk:>5}  ({low_risk / len(customers) * 100:.1f}%)")
    print(f"\nChurn probability stats:")
    print(f"  mean:  {np.mean(churn_probs):.3f}")
    print(f"  std:   {np.std(churn_probs):.3f}")
    print(f"  min:   {np.min(churn_probs):.3f}")
    print(f"  max:   {np.max(churn_probs):.3f}")
    print(f"\nFeature vector dimensions: {len(customers[0]['feature_vector'])}")
    print("--------------------------\n")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    print(f"Generating {NUM_CUSTOMERS} synthetic telecom customers...")
    customers = generate_customers(NUM_CUSTOMERS)

    print_summary(customers)

    confirm = input("Seed these customers to Atlas? (y/n): ")
    if confirm.lower() == "y":
        db = get_db()
        seed_atlas(customers, db)
        print(f"Atlas customers count: {db.customers.count_documents({})}")
    else:
        print("Aborted. No data written to Atlas.")