import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

import mlflow
import mlflow.sklearn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from env.churn_env import ChurnEnv
from data.atlas import get_db, get_all_customers, insert_training_run

load_dotenv()

# --- Hyperparameters ---
# Tune these between runs. Log every change to MLflow so runs are comparable.
HYPERPARAMETERS = {
    "algorithm": "PPO",
    "total_timesteps": 100_000,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "episode_length": 12,
    "n_envs": 4,               # parallel environments — speeds up training
}


# ------------------------------------------------------------------
# Callback — logs metrics to MLflow during training
# ------------------------------------------------------------------

class ChurnMetricsCallback(BaseCallback):
    """
    SB3 callback that logs reward and churn metrics to MLflow
    at regular intervals during training.

    Runs every `log_freq` steps. Computes mean episode reward
    and churn rate over the most recent completed episodes.
    """

    def __init__(self, log_freq: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_churns = []

    def _on_step(self) -> bool:
        # SB3 vectorized envs store episode info in infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
            if "churn_occurred" in info and info["churn_occurred"]:
                self.episode_churns.append(1)
            elif "churn_occurred" in info:
                self.episode_churns.append(0)

        if self.n_calls % self.log_freq == 0 and self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards[-50:])
            churn_rate = np.mean(self.episode_churns[-50:]) if self.episode_churns else 0.0

            mlflow.log_metric("mean_reward", mean_reward, step=self.num_timesteps)
            mlflow.log_metric("churn_rate", churn_rate, step=self.num_timesteps)

            if self.verbose:
                print(
                    f"  step={self.num_timesteps:>7} | "
                    f"mean_reward={mean_reward:+.2f} | "
                    f"churn_rate={churn_rate:.2f}"
                )

        return True


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(hyperparameters: dict = HYPERPARAMETERS) -> dict:
    """
    Train a PPO agent on the ChurnEnv.

    Loads customer pool from Atlas, sets up vectorized environments,
    trains with SB3 PPO, logs everything to MLflow, saves the model,
    and writes a training run summary to Atlas.

    Args:
        hyperparameters: dict of training config. Defaults to HYPERPARAMETERS.

    Returns:
        run_summary: dict written to Atlas training_runs collection
    """
    run_id = str(uuid.uuid4())
    model_path = f"models/ppo_churn_{run_id[:8]}"

    print(f"\n{'='*60}")
    print(f"Churn Intervention RL — Training Run")
    print(f"Run ID: {run_id[:8]}")
    print(f"{'='*60}\n")

    # --- Load customer pool from Atlas ---
    print("Loading customer pool from Atlas...")
    db = get_db()
    customers = get_all_customers(db)
    print(f"  Loaded {len(customers)} customers\n")

    # --- Compute baseline churn rate (no intervention) ---
    baseline_env = ChurnEnv(customers=customers)
    baseline_churns = []
    print("Computing baseline churn rate (random policy, 200 episodes)...")
    for _ in range(200):
        obs, _ = baseline_env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = baseline_env.action_space.sample()
            obs, _, terminated, truncated, _ = baseline_env.step(action)
        summary = baseline_env.get_episode_summary()
        baseline_churns.append(int(summary.get("churned", False)))
    churn_rate_baseline = float(np.mean(baseline_churns))
    print(f"  Baseline churn rate: {churn_rate_baseline:.2%}\n")

    # --- Set up vectorized environments ---
    def make_env():
        return ChurnEnv(
            customers=customers,
            episode_length=hyperparameters["episode_length"],
        )

    vec_env = make_vec_env(make_env, n_envs=hyperparameters["n_envs"])

    # --- MLflow run ---
    mlflow.set_tracking_uri("./mlflow")
    mlflow.set_experiment("churn-intervention-rl")

    with mlflow.start_run(run_name=f"ppo_{run_id[:8]}") as mlflow_run:
        mlflow.log_params(hyperparameters)
        mlflow.log_param("n_customers", len(customers))
        mlflow.log_param("churn_rate_baseline", churn_rate_baseline)

        # --- Build PPO model ---
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=hyperparameters["learning_rate"],
            n_steps=hyperparameters["n_steps"],
            batch_size=hyperparameters["batch_size"],
            n_epochs=hyperparameters["n_epochs"],
            gamma=hyperparameters["gamma"],
            verbose=0,
        )

        callback = ChurnMetricsCallback(
            log_freq=hyperparameters["n_steps"],
            verbose=1,
        )

        # --- Train ---
        print(f"Training PPO for {hyperparameters['total_timesteps']:,} timesteps...")
        print(f"  {hyperparameters['n_envs']} parallel environments\n")

        model.learn(
            total_timesteps=hyperparameters["total_timesteps"],
            callback=callback,
            progress_bar=True,
        )

        # --- Save model ---
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        mlflow.log_artifact(model_path + ".zip")
        print(f"\nModel saved: {model_path}.zip")

        # --- Evaluate trained model ---
        print("\nEvaluating trained policy (200 episodes)...")
        eval_env = ChurnEnv(customers=customers)
        eval_churns = []
        eval_rewards = []
        action_counts = {i: 0 for i in range(5)}

        for _ in range(200):
            obs, _ = eval_env.reset()
            terminated = truncated = False
            episode_reward = 0.0
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(int(action))
                episode_reward += reward
                action_counts[int(action)] += 1
            summary = eval_env.get_episode_summary()
            eval_churns.append(int(summary.get("churned", False)))
            eval_rewards.append(episode_reward)

        churn_rate_trained = float(np.mean(eval_churns))
        mean_eval_reward = float(np.mean(eval_rewards))
        total_actions = sum(action_counts.values())
        intervention_distribution = {
            str(i): round(action_counts[i] / total_actions, 4)
            for i in range(5)
        }

        mlflow.log_metric("churn_rate_trained", churn_rate_trained)
        mlflow.log_metric("churn_rate_reduction",
                          churn_rate_baseline - churn_rate_trained)
        mlflow.log_metric("mean_eval_reward", mean_eval_reward)

        print(f"  Trained churn rate:   {churn_rate_trained:.2%}")
        print(f"  Baseline churn rate:  {churn_rate_baseline:.2%}")
        print(f"  Reduction:            {churn_rate_baseline - churn_rate_trained:.2%}")
        print(f"  Mean eval reward:     {mean_eval_reward:+.2f}")
        print(f"\n  Action distribution (trained policy):")
        from env.reward import action_name
        for i, pct in intervention_distribution.items():
            print(f"    {action_name(int(i)):<24} {pct:.1%}")

        # --- Build reward curve from callback ---
        reward_curve = [
            {
                "step": i * hyperparameters["n_steps"],
                "mean_reward": float(r),
            }
            for i, r in enumerate(callback.episode_rewards[::10])
        ]

        # --- Write training run to Atlas ---
        run_summary = {
            "run_id": run_id,
            "mlflow_run_id": mlflow_run.info.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "algorithm": hyperparameters["algorithm"],
            "hyperparameters": hyperparameters,
            "n_customers": len(customers),
            "churn_rate_baseline": churn_rate_baseline,
            "churn_rate_trained": churn_rate_trained,
            "churn_rate_reduction": churn_rate_baseline - churn_rate_trained,
            "mean_eval_reward": mean_eval_reward,
            "reward_curve": reward_curve,
            "intervention_distribution": intervention_distribution,
            "model_path": model_path + ".zip",
        }

        insert_training_run(run_summary, db)
        print(f"\nTraining run written to Atlas (run_id: {run_id[:8]})")
        print(f"MLflow run: {mlflow_run.info.run_id}")
        print(f"\n{'='*60}\n")

        return run_summary


if __name__ == "__main__":
    summary = train()
    print("Done.")