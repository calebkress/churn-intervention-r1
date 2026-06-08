import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


PLAN_TYPES = ["prepaid", "postpaid", "enterprise"]
PLAN_TYPE_ENCODING = {"prepaid": 0, "postpaid": 1, "enterprise": 2}


@dataclass
class Customer:
    customer_id: str
    tenure_months: int
    plan_type: str
    monthly_spend: float
    contract_end_date: str
    avg_monthly_data_gb: float
    call_drop_rate: float
    support_tickets_90d: int
    payment_failures_90d: int
    nps_score: int
    days_since_last_contact: int
    churn_probability: float
    churn_label: Optional[bool] = None
    feature_vector: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "tenure_months": self.tenure_months,
            "plan_type": self.plan_type,
            "monthly_spend": self.monthly_spend,
            "contract_end_date": self.contract_end_date,
            "avg_monthly_data_gb": self.avg_monthly_data_gb,
            "call_drop_rate": self.call_drop_rate,
            "support_tickets_90d": self.support_tickets_90d,
            "payment_failures_90d": self.payment_failures_90d,
            "nps_score": self.nps_score,
            "days_since_last_contact": self.days_since_last_contact,
            "churn_probability": self.churn_probability,
            "churn_label": self.churn_label,
            "feature_vector": self.feature_vector,
        }

    def to_observation(self, interventions_this_episode: int, last_action: int, steps_remaining: int, episode_length: int = 12) -> np.ndarray:
        """
        Returns a normalized float32 observation vector for the RL agent.
        Shape: (13,)
        """
        risk_level = 0.0 if self.churn_probability < 0.3 else (0.5 if self.churn_probability < 0.6 else 1.0)
        
        return np.array([
            self.tenure_months / 120.0,
            PLAN_TYPE_ENCODING[self.plan_type] / 2.0,
            self.monthly_spend / 300.0,
            self.avg_monthly_data_gb / 50.0,
            self.call_drop_rate,
            self.support_tickets_90d / 10.0,
            self.payment_failures_90d / 5.0,
            self.nps_score / 10.0,
            self.days_since_last_contact / 180.0,
            self.churn_probability,
            interventions_this_episode / 10.0,
            last_action / 4.0,
            steps_remaining / episode_length,
            risk_level,
        ], dtype=np.float32)


def compute_churn_probability(customer: "Customer") -> float:
    """
    Derive churn probability from customer features.
    Weights based on relative signal strength of each feature.
    """
    raw = (
        0.30 * min(customer.support_tickets_90d / 10.0, 1.0)
        + 0.25 * min(customer.call_drop_rate, 1.0)
        + 0.20 * (1.0 - customer.nps_score / 10.0)
        + 0.15 * min(customer.payment_failures_90d / 5.0, 1.0)
        + 0.10 * (1.0 - min(customer.tenure_months, 60) / 60.0)
    )
    noise = np.random.normal(0, 0.05)
    return float(np.clip(raw + noise, 0.0, 1.0))


def compute_feature_vector(customer: "Customer") -> list:
    """
    Normalized behavioral feature vector for Atlas Vector Search.
    10 dimensions, all values in [0, 1].
    """
    return [
        min(customer.tenure_months / 120.0, 1.0),
        PLAN_TYPE_ENCODING[customer.plan_type] / 2.0,
        min(customer.monthly_spend / 300.0, 1.0),
        min(customer.avg_monthly_data_gb / 50.0, 1.0),
        min(customer.call_drop_rate, 1.0),
        min(customer.support_tickets_90d / 10.0, 1.0),
        min(customer.payment_failures_90d / 5.0, 1.0),
        customer.nps_score / 10.0,
        min(customer.days_since_last_contact / 180.0, 1.0),
        customer.churn_probability,
    ]


def resolve_churn(customer: "Customer", action: int) -> bool:
    """
    Stochastically resolve whether a customer churns this timestep.

    Intervention effectiveness varies by action and customer segment.
    Each action has a base retention boost applied to churn probability
    before sampling the Bernoulli outcome.

    Args:
        customer: current customer state
        action: action taken this timestep (0-4)

    Returns:
        True if customer churned, False if retained
    """
    # Base effectiveness of each intervention at reducing churn probability
    # Indexed by action: [do_nothing, email, call, discount, escalate]
    base_effectiveness = [0.0, 0.05, 0.12, 0.18, 0.25]

    # Enterprise customers respond better to escalation and calls
    # Prepaid customers respond better to discounts
    segment_modifier = 0.0
    if customer.plan_type == "enterprise":
        if action in [2, 4]:  # call or escalate
            segment_modifier = 0.08
    elif customer.plan_type == "prepaid":
        if action == 3:  # discount
            segment_modifier = 0.06

    # High-tenure customers are harder to lose — small baseline retention boost
    tenure_modifier = min(customer.tenure_months / 120.0, 1.0) * 0.05

    effective_churn_prob = customer.churn_probability - base_effectiveness[action] - segment_modifier - tenure_modifier
    effective_churn_prob = float(np.clip(effective_churn_prob, 0.0, 1.0))

    return bool(np.random.random() < effective_churn_prob)


def generate_customer() -> Customer:
    """
    Generate a single synthetic telecom customer with realistic distributions.
    Computes churn probability and feature vector automatically.
    """
    plan_type = np.random.choice(
        PLAN_TYPES,
        p=[0.40, 0.45, 0.15]
    )

    tenure_months = int(np.random.exponential(scale=24))
    tenure_months = np.clip(tenure_months, 1, 120)

    # Spend correlated with plan type
    if plan_type == "prepaid":
        monthly_spend = float(np.random.normal(35, 10))
    elif plan_type == "postpaid":
        monthly_spend = float(np.random.normal(75, 20))
    else:  # enterprise
        monthly_spend = float(np.random.normal(180, 50))
    monthly_spend = max(10.0, monthly_spend)

    avg_monthly_data_gb = float(np.random.exponential(scale=8))
    avg_monthly_data_gb = np.clip(avg_monthly_data_gb, 0.5, 50.0)

    call_drop_rate = float(np.random.beta(1.5, 10))  # skewed low, occasional spikes
    support_tickets_90d = int(np.random.poisson(1.2))
    support_tickets_90d = min(support_tickets_90d, 10)

    payment_failures_90d = int(np.random.poisson(0.3))
    payment_failures_90d = min(payment_failures_90d, 5)

    nps_score = int(np.random.choice(range(0, 11), p=[
        0.05, 0.05, 0.05, 0.05, 0.08, 0.10, 0.12, 0.15, 0.15, 0.10, 0.10
    ]))

    days_since_last_contact = int(np.random.exponential(scale=30))
    days_since_last_contact = np.clip(days_since_last_contact, 0, 180)

    # Placeholder — will be computed after instantiation
    customer = Customer(
        customer_id=str(uuid.uuid4()),
        tenure_months=int(tenure_months),
        plan_type=plan_type,
        monthly_spend=round(monthly_spend, 2),
        contract_end_date="2027-01-01",
        avg_monthly_data_gb=round(float(avg_monthly_data_gb), 2),
        call_drop_rate=round(float(call_drop_rate), 4),
        support_tickets_90d=int(support_tickets_90d),
        payment_failures_90d=int(payment_failures_90d),
        nps_score=int(nps_score),
        days_since_last_contact=int(days_since_last_contact),
        churn_probability=0.0,
    )

    customer.churn_probability = round(compute_churn_probability(customer), 4)
    customer.feature_vector = compute_feature_vector(customer)

    return customer


if __name__ == "__main__":
    # Smoke test
    np.random.seed(42)
    customer = generate_customer()
    print("Generated customer:")
    for k, v in customer.to_dict().items():
        if k != "feature_vector":
            print(f"  {k}: {v}")
    print(f"  feature_vector: [{', '.join(f'{x:.3f}' for x in customer.feature_vector)}]")

    print("\nObservation vector (13 dims):")
    obs = customer.to_observation(interventions_this_episode=0, last_action=0, steps_remaining=12)
    print(f"  {obs}")

    print("\nChurn resolution (10 trials, action=outbound_call):")
    results = [resolve_churn(customer, action=2) for _ in range(10)]
    print(f"  churned: {sum(results)}/10 (churn_prob={customer.churn_probability})")