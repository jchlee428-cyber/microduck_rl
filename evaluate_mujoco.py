"""
MicroDuck RL Lab - MuJoCo Hopper Evaluation Script
Evaluates trained PPO model on Hopper-v5 and compares with random actions.
"""
import _bootstrap  # Automatically redirect to .venv python
import os
import argparse
import numpy as np
import mujoco_compat  # Windows Unicode path compatibility
import microduck_env  # Custom MicroDuck-v1 environment
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def evaluate(
    env_id: str = "Hopper-v5",
    model_path: str = "models/hopper_ppo.zip",
    vec_norm_path: str = "models/hopper_vec_normalize.pkl",
    num_episodes: int = 5,
    render: bool = False,
    compare_random: bool = True
):
    print("=" * 60)
    print(f" MicroDuck RL Lab: MuJoCo Robot Evaluation ({env_id})")
    print("=" * 60)

    render_mode = "human" if render else None

    # 1. Evaluate Random Agent (Baseline)
    if compare_random:
        print("\n--- 1. Evaluating Random Agent (Untrained Baseline) ---")
        env_random = gym.make(env_id, render_mode=render_mode)
        random_rewards = []
        random_steps_list = []

        for ep in range(1, num_episodes + 1):
            obs, info = env_random.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = env_random.action_space.sample()
                obs, reward, terminated, truncated, info = env_random.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            random_rewards.append(total_reward)
            random_steps_list.append(steps)
            print(f"  [Random] Episode {ep}: Total Reward = {total_reward:.2f}, Survival Steps = {steps}")

        env_random.close()
        print(f"  --> Random Mean Reward: {np.mean(random_rewards):.2f} +/- {np.std(random_rewards):.2f}")
        print(f"  --> Random Mean Steps : {np.mean(random_steps_list):.1f} steps (Fell down quickly)")

    # 2. Evaluate Trained PPO Agent
    print(f"\n--- 2. Evaluating Trained PPO Model: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at '{model_path}'. Please run train_mujoco.py first.")
        return

    # Load Vectorized Environment & Normalization Stats if available
    def _make_eval_env():
        return gym.make(env_id, render_mode=render_mode)

    eval_vec_env = DummyVecEnv([_make_eval_env])
    if os.path.exists(vec_norm_path):
        print(f"[INFO] Loading normalization statistics from {vec_norm_path}")
        eval_vec_env = VecNormalize.load(vec_norm_path, eval_vec_env)
        eval_vec_env.training = False
        eval_vec_env.norm_reward = False
    else:
        print("[WARN] VecNormalize stats not found. Evaluating with raw observations.")

    model = PPO.load(model_path, env=eval_vec_env)
    trained_rewards = []
    trained_steps_list = []

    for ep in range(1, num_episodes + 1):
        obs = eval_vec_env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_vec_env.step(action)
            # When using VecEnv, reward is an array and done is dones[0]
            total_reward += float(reward[0])
            steps += 1
            done = bool(dones[0])

        trained_rewards.append(total_reward)
        trained_steps_list.append(steps)
        print(f"  [Trained PPO] Episode {ep}: Total Reward = {total_reward:.2f}, Survival Steps = {steps}")

    eval_vec_env.close()

    print("\n" + "=" * 60)
    print(f" EVALUATION SUMMARY (MuJoCo {env_id})")
    print("=" * 60)
    if compare_random:
        print(f"  Random Agent Average Reward  : {np.mean(random_rewards):.2f} (Steps: {np.mean(random_steps_list):.1f})")
    print(f"  Trained PPO Average Reward   : {np.mean(trained_rewards):.2f} (Steps: {np.mean(trained_steps_list):.1f})")
    if compare_random:
        reward_diff = np.mean(trained_rewards) - np.mean(random_rewards)
        step_diff = np.mean(trained_steps_list) - np.mean(random_steps_list)
        print(f"  Reward Improvement           : +{reward_diff:.2f}")
        print(f"  Survival Duration Improvement: +{step_diff:.1f} steps")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MuJoCo Robot PPO Model")
    parser.add_argument("--env", type=str, default="Hopper-v5", help="Gymnasium MuJoCo Environment ID (e.g. Hopper-v5, Walker2d-v5)")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model")
    parser.add_argument("--vec-norm", type=str, default=None, help="Path to VecNormalize file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render simulator window")
    parser.add_argument("--no-compare", action="store_true", help="Skip random agent comparison")
    args = parser.parse_args()

    name = args.env.split("-")[0].lower()
    model_path = args.model or f"models/{name}_ppo.zip"
    vec_norm_path = args.vec_norm or f"models/{name}_vec_normalize.pkl"

    evaluate(
        env_id=args.env,
        model_path=model_path,
        vec_norm_path=vec_norm_path,
        num_episodes=args.episodes,
        render=args.render,
        compare_random=not args.no_compare
    )
