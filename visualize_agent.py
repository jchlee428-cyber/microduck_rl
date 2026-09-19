"""
MicroDuck RL Lab - Agent Visualization & Video Recording Script
Visualizes or records comparing Random actions vs Trained PPO actions.
"""
import _bootstrap  # Automatically redirect to .venv python
import os
import time
import argparse
import imageio
import numpy as np
import mujoco_compat  # Windows Unicode path compatibility
import microduck_env  # Custom MicroDuck-v1 environment
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def record_comparison(
    env_id: str = "Hopper-v5",
    model_path: str = "models/hopper_ppo.zip",
    vec_norm_path: str = "models/hopper_vec_normalize.pkl",
    output_gif: str = "videos/hopper_comparison.gif",
    max_steps: int = 300,
    fps: int = 30
):
    print("=" * 60)
    print(f" Recording Visual Comparison: Random vs Trained ({env_id})")
    print("=" * 60)
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)

    # 1. Record Random Agent
    print("[1/3] Recording Random Agent (Untrained)...")
    env_random = gym.make(env_id, render_mode="rgb_array")
    obs, info = env_random.reset()
    random_frames = []
    
    for step in range(max_steps):
        frame = env_random.render()
        random_frames.append(frame)
        action = env_random.action_space.sample()
        obs, reward, terminated, truncated, info = env_random.step(action)
        if terminated or truncated:
            # Hold last frame briefly to show the fall
            for _ in range(15):
                random_frames.append(frame)
            break
    env_random.close()
    print(f"      Captured {len(random_frames)} frames for Random Agent.")

    # 2. Record Trained PPO Agent
    print(f"[2/3] Recording Trained PPO Agent from {model_path}...")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}.")
        return

    def _make_env():
        return gym.make(env_id, render_mode="rgb_array")

    eval_env = DummyVecEnv([_make_env])
    if os.path.exists(vec_norm_path):
        eval_env = VecNormalize.load(vec_norm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env)
    obs = eval_env.reset()
    trained_frames = []

    for step in range(max_steps):
        frame = eval_env.envs[0].render()
        trained_frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = eval_env.step(action)
        if dones[0]:
            for _ in range(15):
                trained_frames.append(frame)
            break
    eval_env.close()
    print(f"      Captured {len(trained_frames)} frames for Trained Agent.")

    # 3. Save as GIF
    print(f"[3/3] Saving combined animation to {output_gif}...")
    all_frames = random_frames + trained_frames
    # Sample every 2 frames for compact file size and smooth playback
    subsampled = all_frames[::2]
    imageio.mimsave(output_gif, subsampled, fps=fps // 2, loop=0)
    print(f"[SUCCESS] Comparison animation saved: {output_gif} ({len(subsampled)} frames)")
    print("=" * 60)

def play_human_window(
    env_id: str = "Hopper-v5",
    model_path: str = None,
    vec_norm_path: str = None,
    episodes: int = 10,
    loop: bool = False
):
    # Auto configure model paths if not specified
    if model_path is None:
        if "CartPole" in env_id:
            model_path = "models/cartpole_ppo.zip"
            vec_norm_path = None
        else:
            name = env_id.split("-")[0].lower()
            model_path = f"models/{name}_ppo.zip"
            vec_norm_path = f"models/{name}_vec_normalize.pkl"

    print("=" * 60)
    print(f" Opening Real-time Simulation Window: {env_id}")
    print(f" Model: {model_path}")
    print(" (Press Ctrl+C in terminal or close window to stop)")
    print("=" * 60)

    try:
        def _make_env():
            return gym.make(env_id, render_mode="human")

        eval_env = DummyVecEnv([_make_env])
        if vec_norm_path and os.path.exists(vec_norm_path):
            eval_env = VecNormalize.load(vec_norm_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False

        model = PPO.load(model_path, env=eval_env)

        ep = 1
        while loop or (ep <= episodes):
            obs = eval_env.reset()
            done = False
            total_rew = 0.0
            steps = 0
            print(f"[Episode {ep}] Running simulation...")
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, info = eval_env.step(action)
                total_rew += float(reward[0])
                steps += 1
                done = bool(dones[0])
                time.sleep(0.015)  # Natural playback speed

            print(f"  --> Episode {ep} Finished: Total Reward = {total_rew:.1f}, Steps = {steps}")
            time.sleep(0.5)  # Pause briefly before next episode
            ep += 1

        eval_env.close()
        print("\n[SUCCESS] Simulation completed.")
    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped by user.")
    except Exception as e:
        print(f"\n[WARN] Failed to open human render window: {e}")
        print("[TIP] You can still view the saved GIF/video in the 'videos/' folder!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and Record RL Agent")
    parser.add_argument("--env", type=str, default="Hopper-v5", help="Gymnasium environment ID (e.g. Hopper-v5, CartPole-v1)")
    parser.add_argument("--model", type=str, default=None, help="Trained model path")
    parser.add_argument("--vec-norm", type=str, default=None, help="VecNormalize path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to show (default: 10)")
    parser.add_argument("--loop", action="store_true", help="Run simulation in an infinite loop until stopped")
    parser.add_argument("--human", action="store_true", help="Display live simulation window")
    parser.add_argument("--output", type=str, default="videos/hopper_comparison.gif", help="Output GIF path")
    args = parser.parse_args()

    if args.human:
        play_human_window(
            env_id=args.env,
            model_path=args.model,
            vec_norm_path=args.vec_norm,
            episodes=args.episodes,
            loop=args.loop
        )
    else:
        name = args.env.split("-")[0].lower()
        default_output = f"videos/{name}_comparison.gif"
        output_path = args.output if args.output != "videos/hopper_comparison.gif" else default_output
        record_comparison(
            env_id=args.env,
            model_path=args.model or f"models/{name}_ppo.zip",
            vec_norm_path=args.vec_norm or f"models/{name}_vec_normalize.pkl",
            output_gif=output_path
        )
