"""
reward.py — Churn Intervention RL Reward Function

Tune reward weights here only. Do not hardcode reward logic in churn_env.py.

Design philosophy:
  The agent should learn to retain customers profitably without burning
  relationships through over-intervention or repetitive contact. A good policy:
    - Intervenes appropriately on high-risk customers
    - Varies its approach rather than repeating the same action
    - Leaves low-risk customers alone (do_nothing is a positive signal)
    - Doesn't spam expensive interventions indiscriminately
    - Backs off when a customer has been contacted too many times
"""

# --- Action costs ---
# Indexed by action: [do_nothing, email, call, discount, escalate]
ACTION_COSTS = [0.0, 0.5, 2.0, 5.0, 8.0]

# --- Reward weights ---
RETENTION_BONUS = 15.0        # reward for successfully retaining a customer
CHURN_PENALTY = -20.0         # penalty when a customer churns
OVER_CONTACT_THRESHOLD = 3    # max interventions before penalty kicks in
OVER_CONTACT_PENALTY = -3.0   # penalty per step above threshold
EFFICIENCY_BONUS = 1.5        # reward for do_nothing on a low-risk customer
LOW_RISK_THRESHOLD = 0.20     # churn probability below which do_nothing is rewarded
REPETITION_PENALTY = -2.0     # penalty for taking same non-zero action 3+ times in a row


def compute_reward(
    action: int,
    churn_occurred: bool,
    churn_probability: float,
    interventions_this_episode: int,
    last_actions: list = [],
) -> tuple[float, dict]:
    """
    Compute the reward for a single environment step.

    Args:
        action: action taken this step (0-4)
        churn_occurred: whether the customer churned this step
        churn_probability: customer's current churn probability
        interventions_this_episode: total interventions taken so far this episode
        last_actions: list of recent actions taken this episode

    Returns:
        total_reward: float
        breakdown: dict of reward components for logging and debugging
    """
    retention_bonus = RETENTION_BONUS if not churn_occurred else 0.0
    churn_penalty = CHURN_PENALTY if churn_occurred else 0.0
    intervention_cost = -ACTION_COSTS[action]

    # Penalize over-contacting
    over_contact_penalty = (
        OVER_CONTACT_PENALTY
        if action != 0 and interventions_this_episode >= OVER_CONTACT_THRESHOLD
        else 0.0
    )

    # Reward doing nothing on genuinely low-risk customers
    efficiency_bonus = (
        EFFICIENCY_BONUS
        if action == 0 and churn_probability < LOW_RISK_THRESHOLD
        else 0.0
    )

    # Penalize repeating the same non-zero action 3+ times in a row
    repetition_penalty = (
        REPETITION_PENALTY
        if action != 0
        and len(last_actions) >= 2
        and last_actions[-1] == action
        and last_actions[-2] == action
        else 0.0
    )

    total_reward = (
        retention_bonus
        + churn_penalty
        + intervention_cost
        + over_contact_penalty
        + efficiency_bonus
        + repetition_penalty
    )

    breakdown = {
        "retention_bonus": retention_bonus,
        "churn_penalty": churn_penalty,
        "intervention_cost": intervention_cost,
        "over_contact_penalty": over_contact_penalty,
        "efficiency_bonus": efficiency_bonus,
        "repetition_penalty": repetition_penalty,
        "total": total_reward,
    }

    return total_reward, breakdown


def action_name(action: int) -> str:
    """Human-readable action label. Used in logging and dashboard."""
    names = [
        "do_nothing",
        "send_email_offer",
        "outbound_call",
        "discount_10pct",
        "escalate_to_retention",
    ]
    return names[action]


# --- Tuning guide ---
#
# If agent always does nothing:
#   CHURN_PENALTY is too weak relative to ACTION_COSTS.
#   Try increasing CHURN_PENALTY or decreasing ACTION_COSTS across the board.
#
# If agent spams one intervention on every customer:
#   REPETITION_PENALTY is too weak. Increase it.
#   Or ACTION_COSTS are too uniform — widen the gap between cheap and expensive.
#
# If agent spams escalate_to_retention on every customer:
#   ACTION_COSTS[4] is too low or RETENTION_BONUS is too high.
#   Try increasing ACTION_COSTS[4] or reducing RETENTION_BONUS.
#
# If agent ignores low-risk customers but still contacts them occasionally:
#   EFFICIENCY_BONUS is too weak. Increase it, or lower LOW_RISK_THRESHOLD.
#
# If agent over-contacts high-risk customers repeatedly:
#   Lower OVER_CONTACT_THRESHOLD from 3 to 2, or increase OVER_CONTACT_PENALTY.
#
# General principle: the ratio of CHURN_PENALTY to ACTION_COSTS determines
# how aggressive the agent is. A 4:1 ratio (20 penalty, 5 max cost) produces
# moderate intervention behavior. Widen the ratio for more aggression,
# narrow it for more restraint.


if __name__ == "__main__":
    print("=== Reward function smoke test ===\n")

    scenarios = [
        {
            "label": "Do nothing, customer retained, low risk",
            "action": 0, "churn_occurred": False,
            "churn_probability": 0.10, "interventions_this_episode": 0,
            "last_actions": [],
        },
        {
            "label": "Escalate, customer retained, high risk",
            "action": 4, "churn_occurred": False,
            "churn_probability": 0.80, "interventions_this_episode": 1,
            "last_actions": [2],
        },
        {
            "label": "Escalate, customer churns anyway",
            "action": 4, "churn_occurred": True,
            "churn_probability": 0.80, "interventions_this_episode": 1,
            "last_actions": [2],
        },
        {
            "label": "Email, customer retained, over-contact",
            "action": 1, "churn_occurred": False,
            "churn_probability": 0.50, "interventions_this_episode": 4,
            "last_actions": [2, 3],
        },
        {
            "label": "Outbound call, third time in a row (repetition penalty)",
            "action": 2, "churn_occurred": False,
            "churn_probability": 0.60, "interventions_this_episode": 2,
            "last_actions": [2, 2],
        },
        {
            "label": "Do nothing, customer churns, high risk",
            "action": 0, "churn_occurred": True,
            "churn_probability": 0.75, "interventions_this_episode": 0,
            "last_actions": [],
        },
    ]

    for s in scenarios:
        reward, breakdown = compute_reward(
            action=s["action"],
            churn_occurred=s["churn_occurred"],
            churn_probability=s["churn_probability"],
            interventions_this_episode=s["interventions_this_episode"],
            last_actions=s["last_actions"],
        )
        print(f"Scenario: {s['label']}")
        print(f"  Action: {action_name(s['action'])}")
        for k, v in breakdown.items():
            if k != "total" and v != 0.0:
                print(f"  {k}: {v:+.1f}")
        print(f"  TOTAL: {reward:+.1f}\n")