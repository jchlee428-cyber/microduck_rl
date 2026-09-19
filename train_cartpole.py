"""
MicroDuck RL Lab - CartPole Training Script
Train a PPO agent on Gymnasium CartPole-v1 using Stable-Baselines3.
"""
import _bootstrap  # Automatically redirect to .venv python
import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

def train():
    print("=" * 60)
    print(" MicroDuck RL Lab: CartPole-v1 PPO Training")
    print("=" * 60)
    
    # 1. Paths setup
    models_dir = os.path.join(".", "models")
    logs_dir = os.path.join(".", "logs", "cartpole")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 2. Environment creation (with Monitor wrapper for logging episode stats)
    env = gym.make("CartPole-v1")
    monitored_env = Monitor(env)

    # 3. Evaluation environment & callback
    eval_env = Monitor(gym.make("CartPole-v1"))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best_cartpole"),
        log_path=logs_dir,
        eval_freq=2000,
        deterministic=True,
        render=False,
        verbose=1
    )

    # 4. PPO Agent initialization
    # Using MlpPolicy (Multi-Layer Perceptron) for vector observations
    model = PPO(
        policy="MlpPolicy",
        env=monitored_env,
        learning_rate=0.001,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log=logs_dir
    )

    total_timesteps = 20_000
    print(f"\n[INFO] Starting training for {total_timesteps:,} timesteps...")
    print(f"[INFO] Policy: MlpPolicy | Environment: CartPole-v1")
    print(f"[INFO] Tensorboard logs: {logs_dir}")
    print("[INFO] Note: Training time may vary depending on your CPU/GPU performance.\n")

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    elapsed_time = time.time() - start_time

    # 5. Save final model
    final_model_path = os.path.join(models_dir, "cartpole_ppo.zip")
    model.save(final_model_path)
    
    env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print(f" TRAINING COMPLETE!")
    print(f" Total training time: {elapsed_time:.2f} seconds")
    print(f" Final model saved to: {final_model_path}")
    print(f" Best model saved to : {os.path.join(models_dir, 'best_cartpole', 'best_model.zip')}")
    print("=" * 60)

if __name__ == "__main__":
    train()
