import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

from stable_baselines3 import PPO

from env.churn_env import ChurnEnv
from env.reward import action_name
from data.atlas import (
    get_db,
    get_all_customers,
    insert_interventions_bulk,
    update_churn_label,
)

load_dotenv()

# --- Configuration ---
N_EVAL_EPISODES = 500       # number of episodes to evaluate
MODEL_PATH = None           # set to a specific path to override auto-detection


def find_latest_model(models_dir: str = "models") -> str:
    """
    Find the most recently created model file in the models directory.
    Returns the path without the .zip extension (SB3 convention).
    """
    zips = [
        f for f in os.listdir(models_dir)
        if f.endswith(".zip") and f.startswith("ppo_churn_")
    ]
    if not zips:
        raise FileNotFoundError(f"No model files found in {models_dir}/")

    latest = sorted(
        zips,
        key=lambda f: os.path.getmtime(os.path.join(models_dir, f)),
        reverse=True,
    )[0]

    path = os.path.join(models_dir, latest.replace(".zip", ""))
    print(f"  Using model: {latest}")
    return path


def evaluate(model_path: str = None, n_episodes: int = N_EVAL_EPISODES) -> dict:
    """
    Run the trained PPO policy against the Atlas customer pool
    and write intervention outcomes to Atlas.

    For each episode:
    - Samples a customer from Atlas
    - Runs the trained policy deterministically
    - Records every intervention taken
    - Records the final churn outcome on the customer document

    Args:
        model_path: path to saved SB3 model (without .zip).
                    If None, uses the most recently trained model.
        n_episodes: number of evaluation episodes to run

    Returns:
        summary dict with aggregate stats
    """
    print(f"\n{'='*60}")
    print(f"Churn Intervention RL — Evaluation")
    print(f"{'='*60}\n")

    # --- Load model ---
    print("Loading model...")
    if model_path is None:
        model_path = find_latest_model()
    model = PPO.load(model_path)
    print(f"  Model loaded\n")

    # --- Load customer pool ---
    print("Loading customer pool from Atlas...")
    db = get_db()
    customers = get_all_customers(db)
    print(f"  Loaded {len(customers)} customers\n")

    # --- Run evaluation ---
    env = ChurnEnv(customers=customers)
    intervention_buffer = []
    churn_outcomes = []
    episode_rewards = []
    action_counts = {i: 0 for i in range(5)}

    print(f"Running {n_episodes} evaluation episodes...")

    for episode in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0.0
        episode_interventions = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            action_counts[action] += 1

            # Record every non-trivial intervention
            if action != 0:
                episode_interventions.append({
                    "intervention_id": str(uuid.uuid4()),
                    "customer_id": info["customer_id"],
                    "type": info["action_name"],
                    "date": datetime.now(timezone.utc).isoformat(),
                    "outcome": "churned" if info["churn_occurred"] else "retained",
                    "agent_action_index": action,
                })

        summary = env.get_episode_summary()
        churned = summary.get("churned", False)
        churn_outcomes.append(int(churned))
        episode_rewards.append(episode_reward)

        # Buffer interventions for bulk write
        intervention_buffer.extend(episode_interventions)

        # Update churn label on customer document
        if summary.get("customer_id"):
            update_churn_label(summary["customer_id"], churned, db)

        # Progress indicator
        if (episode + 1) % 100 == 0:
            churn_so_far = np.mean(churn_outcomes)
            print(f"  Episode {episode + 1}/{n_episodes} | "
                  f"churn_rate={churn_so_far:.2%} | "
                  f"mean_reward={np.mean(episode_rewards):+.2f}")

    # --- Bulk write interventions to Atlas ---
    print(f"\nWriting {len(intervention_buffer)} intervention records to Atlas...")
    insert_interventions_bulk(intervention_buffer, db)
    print(f"  Done")

    # --- Build summary ---
    total_actions = sum(action_counts.values())
    churn_rate = float(np.mean(churn_outcomes))
    mean_reward = float(np.mean(episode_rewards))
    intervention_distribution = {
        action_name(i): round(action_counts[i] / total_actions, 4)
        for i in range(5)
    }

    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"  Episodes:          {n_episodes}")
    print(f"  Churn rate:        {churn_rate:.2%}")
    print(f"  Mean reward:       {mean_reward:+.2f}")
    print(f"  Total interventions written: {len(intervention_buffer)}")
    print(f"\n  Action distribution:")
    for name, pct in intervention_distribution.items():
        print(f"    {name:<24} {pct:.1%}")
    print(f"\n  Interventions written to Atlas: {len(intervention_buffer)}")
    print(f"{'='*60}\n")

    return {
        "n_episodes": n_episodes,
        "churn_rate": churn_rate,
        "mean_reward": mean_reward,
        "intervention_distribution": intervention_distribution,
        "n_interventions_written": len(intervention_buffer),
    }


if __name__ == "__main__":
    summary = evaluate(model_path=MODEL_PATH, n_episodes=N_EVAL_EPISODES)
    print("Done.")