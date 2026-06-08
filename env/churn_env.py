import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from env.customer import Customer, generate_customer, resolve_churn
from env.reward import compute_reward, action_name


class ChurnEnv(gym.Env):
    """
    A custom Gymnasium environment simulating sequential churn intervention
    decisions for a telecom customer population.

    Each episode represents one customer's lifecycle over 12 monthly timesteps.
    At each step the agent observes the customer's current state and selects
    an intervention action. The environment resolves whether the customer
    churns stochastically and returns a shaped reward signal.

    Observation space: Box(13,) — normalized customer features + episode state
    Action space: Discrete(5) — do_nothing, email, call, discount, escalate

    Args:
        customers: optional pre-loaded list of customer dicts from Atlas.
                   If provided, episodes sample from this pool.
                   If None, generates a fresh synthetic customer each episode.
        episode_length: number of monthly timesteps per episode (default 12)
        verbose: if True, prints step-level info for debugging
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        customers: Optional[list[dict]] = None,
        episode_length: int = 12,
        verbose: bool = False,
    ):
        super().__init__()

        self.customers = customers
        self.episode_length = episode_length
        self.verbose = verbose

        # --- Spaces ---
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

        # --- Episode state ---
        self.customer: Optional[Customer] = None
        self.current_step: int = 0
        self.interventions_this_episode: int = 0
        self.last_action: int = 0
        self.episode_reward: float = 0.0
        self.episode_log: list[dict] = []

    # ------------------------------------------------------------------
    # Core Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.
        Samples a customer from the pool or generates a fresh one.
        Returns the initial observation and an empty info dict.
        """
        super().reset(seed=seed)

        self.customer = self._sample_customer()
        self.current_step = 0
        self.interventions_this_episode = 0
        self.last_action = 0
        self.episode_reward = 0.0
        self.episode_log = []

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take one step in the environment.

        Args:
            action: integer 0-4

        Returns:
            observation: np.ndarray shape (13,)
            reward: float
            terminated: bool — True if customer churned
            truncated: bool — True if episode length reached
            info: dict with step metadata for logging
        """
        assert self.customer is not None, "Must call reset() before step()"
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Resolve churn stochastically
        churn_occurred = resolve_churn(self.customer, action)

        # Compute reward
        reward, breakdown = compute_reward(
            action=action,
            churn_occurred=churn_occurred,
            churn_probability=self.customer.churn_probability,
            interventions_this_episode=self.interventions_this_episode,
        )

        # Update episode state
        if action != 0:
            self.interventions_this_episode += 1
        self.last_action = action
        self.current_step += 1
        self.episode_reward += reward

        # Termination conditions
        terminated = churn_occurred
        truncated = self.current_step >= self.episode_length

        # Build info dict
        info = {
            "step": self.current_step,
            "action": action,
            "action_name": action_name(action),
            "churn_occurred": churn_occurred,
            "churn_probability": self.customer.churn_probability,
            "interventions_this_episode": self.interventions_this_episode,
            "reward_breakdown": breakdown,
            "episode_reward": self.episode_reward,
            "customer_id": self.customer.customer_id,
            "plan_type": self.customer.plan_type,
        }

        self.episode_log.append(info)

        if self.verbose:
            self._print_step(info)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        """Rendering is handled by the React dashboard. No-op here."""
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_customer(self) -> Customer:
        """
        Sample a customer for this episode.
        If a pool was provided, sample randomly from it.
        Otherwise generate a fresh synthetic customer.
        """
        if self.customers:
            raw = self.customers[np.random.randint(len(self.customers))]
            return Customer(
                customer_id=raw["customer_id"],
                tenure_months=raw["tenure_months"],
                plan_type=raw["plan_type"],
                monthly_spend=raw["monthly_spend"],
                contract_end_date=raw["contract_end_date"],
                avg_monthly_data_gb=raw["avg_monthly_data_gb"],
                call_drop_rate=raw["call_drop_rate"],
                support_tickets_90d=raw["support_tickets_90d"],
                payment_failures_90d=raw["payment_failures_90d"],
                nps_score=raw["nps_score"],
                days_since_last_contact=raw["days_since_last_contact"],
                churn_probability=raw["churn_probability"],
                churn_label=raw.get("churn_label"),
                feature_vector=raw.get("feature_vector", []),
            )
        return generate_customer()

    def _get_obs(self) -> np.ndarray:
        """Build the current observation vector from customer state."""
        return self.customer.to_observation(
            interventions_this_episode=self.interventions_this_episode,
            last_action=self.last_action,
            steps_remaining=self.episode_length - self.current_step,
            episode_length=self.episode_length,
        )

    def _print_step(self, info: dict) -> None:
        """Verbose step logging for debugging."""
        print(
            f"  Step {info['step']:>2} | "
            f"{info['action_name']:<24} | "
            f"churn_prob={info['churn_probability']:.2f} | "
            f"churned={str(info['churn_occurred']):<5} | "
            f"reward={info['reward_breakdown']['total']:+.1f} | "
            f"ep_reward={info['episode_reward']:+.1f}"
        )

    def get_episode_summary(self) -> dict:
        """
        Returns a summary of the completed episode.
        Call after terminated or truncated is True.
        Useful for logging to Atlas and MLflow.
        """
        if not self.episode_log:
            return {}

        churned = any(step["churn_occurred"] for step in self.episode_log)
        actions_taken = [step["action"] for step in self.episode_log]
        action_distribution = {
            action_name(i): actions_taken.count(i)
            for i in range(5)
        }

        return {
            "customer_id": self.customer.customer_id if self.customer else None,
            "plan_type": self.customer.plan_type if self.customer else None,
            "churn_probability": self.customer.churn_probability if self.customer else None,
            "churned": churned,
            "episode_length": self.current_step,
            "total_reward": self.episode_reward,
            "interventions": self.interventions_this_episode,
            "action_distribution": action_distribution,
        }


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== ChurnEnv smoke test ===\n")

    # Test 1: environment with generated customers
    print("Test 1: Random episode (verbose, generated customer)")
    print("-" * 60)
    env = ChurnEnv(verbose=True)
    obs, info = env.reset(seed=42)
    print(f"Customer: plan={env.customer.plan_type}, "
          f"churn_prob={env.customer.churn_probability:.2f}, "
          f"tenure={env.customer.tenure_months}mo")
    print(f"Initial obs shape: {obs.shape}, dtype: {obs.dtype}\n")

    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    summary = env.get_episode_summary()
    print(f"\nEpisode summary:")
    print(f"  churned:       {summary['churned']}")
    print(f"  steps:         {summary['episode_length']}")
    print(f"  total_reward:  {summary['total_reward']:+.1f}")
    print(f"  interventions: {summary['interventions']}")
    print(f"  actions:       {summary['action_distribution']}")

    # Test 2: observation space compliance
    print("\nTest 2: Observation space compliance")
    print("-" * 60)
    env2 = ChurnEnv()
    obs, _ = env2.reset()
    assert env2.observation_space.contains(obs), "Observation out of bounds!"
    print(f"  obs in bounds: True")
    print(f"  obs range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Test 3: run 100 episodes, report churn rate
    print("\nTest 3: 100 episodes, random policy churn rate")
    print("-" * 60)
    env3 = ChurnEnv()
    churned_count = 0
    total_reward = 0.0
    for _ in range(100):
        obs, _ = env3.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = env3.action_space.sample()
            obs, reward, terminated, truncated, info = env3.step(action)
            total_reward += reward
        if env3.get_episode_summary().get("churned"):
            churned_count += 1

    print(f"  churn rate (random policy): {churned_count}/100")
    print(f"  mean episode reward:        {total_reward / 100:+.1f}")
    print("\nAll tests passed.")