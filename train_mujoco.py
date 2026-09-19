"""
MicroDuck RL Lab - MuJoCo Hopper-v5 PPO Training Script
Trains a PPO agent on continuous control robot benchmark (Hopper-v5).
"""
import _bootstrap  # Automatically redirect to .venv python
import os
import time
import argparse
import mujoco_compat  # Windows Unicode path compatibility
import microduck_env  # Custom MicroDuck-v1 environment
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

def inspect_environment(env_id: str):
    print("=" * 60)
    print(f" Environment Inspection: {env_id}")
    print("=" * 60)
    env = gym.make(env_id)
    print(f"Observation Space : {env.observation_space}")
    print(f"  Shape           : {env.observation_space.shape}")
    print(f"  Low range       : {env.observation_space.low[:4]} ...")
    print(f"  High range      : {env.observation_space.high[:4]} ...")
    print(f"\nAction Space      : {env.action_space}")
    print(f"  Shape           : {env.action_space.shape} (Continuous Joint Torques)")
    print(f"  Low range       : {env.action_space.low}")
    print(f"  High range      : {env.action_space.high}")
    
    obs, info = env.reset()
    print(f"\nInitial Obs sample: {obs[:5]} ... (total {len(obs)} features)")
    env.close()
    print("=" * 60 + "\n")

def make_env(env_id: str, log_dir: str):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env, log_dir)
        return env
    return _init

def train(env_id: str = "Hopper-v5", total_timesteps: int = 50_000):
    inspect_environment(env_id)

    name = env_id.split("-")[0].lower()  # e.g. "hopper", "walker2d"

    # 1. Directories setup
    models_dir = os.path.join(".", "models")
    logs_dir = os.path.join(".", "logs", name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 2. Vectorized & Normalized Environment
    train_vec_env = DummyVecEnv([make_env(env_id, logs_dir)])
    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Evaluation environment with same normalization statistics
    eval_vec_env = DummyVecEnv([make_env(env_id, logs_dir)])
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, training=False)

    # 3. Evaluation Callback
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=os.path.join(models_dir, f"best_{name}"),
        log_path=logs_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )

    # 4. PPO Agent Configuration for Continuous Control
    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log=logs_dir
    )

    print(f"[INFO] Starting PPO training on {env_id} for {total_timesteps:,} timesteps...")
    print(f"[INFO] TensorBoard directory: {logs_dir}")
    print("[INFO] Note: Training continuous robot control takes more computation than CartPole.")
    print("[INFO] Execution speed depends on your CPU/GPU hardware.\n")

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    elapsed_time = time.time() - start_time

    # 5. Save model and VecNormalize statistics
    model_path = os.path.join(models_dir, f"{name}_ppo.zip")
    vec_norm_path = os.path.join(models_dir, f"{name}_vec_normalize.pkl")
    
    model.save(model_path)
    train_vec_env.save(vec_norm_path)

    train_vec_env.close()
    eval_vec_env.close()

    print("\n" + "=" * 60)
    print(f" MUJOCO {env_id.upper()} TRAINING COMPLETE!")
    print(f" Total training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f" Model saved to     : {model_path}")
    print(f" VecNormalize stats : {vec_norm_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on MuJoCo Environment")
    parser.add_argument("--env", type=str, default="Hopper-v5", help="Gymnasium MuJoCo Environment ID (default: Hopper-v5, e.g. Walker2d-v5)")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training timesteps (default: 50,000)")
    args = parser.parse_args()

    train(env_id=args.env, total_timesteps=args.timesteps)
