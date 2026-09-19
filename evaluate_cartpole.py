"""
MicroDuck RL Lab - CartPole Evaluation Script
Evaluates trained PPO model on CartPole-v1 and compares with random actions.
"""
import _bootstrap  # Automatically redirect to .venv python
import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

def evaluate(model_path: str, num_episodes: int = 5, render: bool = False, compare_random: bool = True):
    print("=" * 60)
    print(" MicroDuck RL Lab: CartPole-v1 Evaluation")
    print("=" * 60)

    render_mode = "human" if render else None

    # 1. Evaluate Random Agent (Baseline)
    if compare_random:
        print("\n--- 1. Evaluating Random Agent (Untrained Baseline) ---")
        env_random = gym.make("CartPole-v1", render_mode=render_mode)
        random_rewards = []

        for ep in range(1, num_episodes + 1):
            obs, info = env_random.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = env_random.action_space.sample()  # Random Action
                obs, reward, terminated, truncated, info = env_random.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            random_rewards.append(total_reward)
            print(f"  [Random] Episode {ep}: Total Reward = {total_reward:.1f}, Steps = {steps}")

        env_random.close()
        print(f"  --> Random Agent Mean Reward: {np.mean(random_rewards):.2f} +/- {np.std(random_rewards):.2f}")

    # 2. Evaluate Trained PPO Agent
    print(f"\n--- 2. Evaluating Trained PPO Model: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at '{model_path}'. Please run train_cartpole.py first.")
        return

    model = PPO.load(model_path)
    env_trained = gym.make("CartPole-v1", render_mode=render_mode)
    trained_rewards = []

    for ep in range(1, num_episodes + 1):
        obs, info = env_trained.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_trained.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        trained_rewards.append(total_reward)
        print(f"  [Trained PPO] Episode {ep}: Total Reward = {total_reward:.1f}, Steps = {steps}")

    env_trained.close()

    print("\n" + "=" * 60)
    print(" EVALUATION SUMMARY")
    print("=" * 60)
    if compare_random:
        print(f"  Random Agent Average Reward : {np.mean(random_rewards):.2f}")
    print(f"  Trained PPO Average Reward  : {np.mean(trained_rewards):.2f} (Max possible: 500.0)")
    if compare_random:
        improvement = (np.mean(trained_rewards) - np.mean(random_rewards)) / (np.mean(random_rewards) + 1e-8) * 100
        print(f"  Performance Improvement     : +{improvement:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CartPole PPO Model")
    parser.add_argument("--model", type=str, default="models/cartpole_ppo.zip", help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render simulator window")
    parser.add_argument("--no-compare", action="store_true", help="Skip random agent comparison")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        num_episodes=args.episodes,
        render=args.render,
        compare_random=not args.no_compare
    )
