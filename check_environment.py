"""
MicroDuck RL Lab - Environment Verification Script
Verifies Python version, PyTorch (CUDA if available), Gymnasium, MuJoCo, and Stable-Baselines3.
"""
import _bootstrap  # Automatically redirect to .venv python if called from global python
import sys
import platform
import mujoco_compat  # Windows Unicode path patch for MuJoCo

def check_system():
    print("=" * 60)
    print(" 1. System & Python Environment Check")
    print("=" * 60)
    print(f"OS Platform      : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Python Version   : {sys.version.split()[0]}")
    print(f"Python Executable: {sys.executable}")
    print()

def check_libraries():
    print("=" * 60)
    print(" 2. Core Libraries Import Check")
    print("=" * 60)
    
    # PyTorch Check
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU only"
        print(f"[OK] PyTorch Version         : {torch.__version__} (CUDA: {cuda_avail}, Device: {device_name})")
    except ImportError as e:
        print(f"[FAIL] PyTorch Import Error   : {e}")
        return False

    # Gymnasium Check
    try:
        import gymnasium as gym
        print(f"[OK] Gymnasium Version       : {gym.__version__}")
    except ImportError as e:
        print(f"[FAIL] Gymnasium Import Error : {e}")
        return False

    # Stable-Baselines3 Check
    try:
        import stable_baselines3 as sb3
        print(f"[OK] Stable-Baselines3 Ver   : {sb3.__version__}")
    except ImportError as e:
        print(f"[FAIL] SB3 Import Error       : {e}")
        return False

    # MuJoCo Check
    try:
        import mujoco
        print(f"[OK] MuJoCo Version          : {mujoco.__version__}")
    except ImportError as e:
        print(f"[FAIL] MuJoCo Import Error    : {e}")
        return False
        
    print()
    return True

def test_gym_environments():
    print("=" * 60)
    print(" 3. Gymnasium Environment Initialization Check")
    print("=" * 60)
    import gymnasium as gym

    # CartPole-v1 Test
    try:
        cartpole_env = gym.make("CartPole-v1")
        obs, info = cartpole_env.reset()
        action = cartpole_env.action_space.sample()
        obs, reward, terminated, truncated, info = cartpole_env.step(action)
        cartpole_env.close()
        print(f"[OK] CartPole-v1 initialized & stepped successfully!")
        print(f"     Observation space: {cartpole_env.observation_space}")
        print(f"     Action space     : {cartpole_env.action_space}")
    except Exception as e:
        print(f"[FAIL] CartPole-v1 test failed: {e}")
        return False

    # MuJoCo Hopper-v5 / InvertedPendulum-v5 Test
    try:
        # Gymnasium 1.x uses v5 for MuJoCo environments
        mujoco_env = gym.make("Hopper-v5")
        obs, info = mujoco_env.reset()
        action = mujoco_env.action_space.sample()
        obs, reward, terminated, truncated, info = mujoco_env.step(action)
        mujoco_env.close()
        print(f"[OK] MuJoCo (Hopper-v5) initialized & stepped successfully!")
        print(f"     Observation space: {mujoco_env.observation_space}")
        print(f"     Action space     : {mujoco_env.action_space}")
    except Exception as e:
        print(f"[FAIL] MuJoCo Hopper-v5 test failed: {e}")
        return False

    print()
    return True

def main():
    check_system()
    libs_ok = check_libraries()
    if not libs_ok:
        print("[ERROR] Core libraries check failed. Please review errors.")
        sys.exit(1)
        
    envs_ok = test_gym_environments()
    if not envs_ok:
        print("[ERROR] Gymnasium/MuJoCo environments test failed.")
        sys.exit(1)

    print("=" * 60)
    print(" SUCCESS: All environment checks passed without errors!")
    print(" MicroDuck RL Lab is ready for reinforcement learning training.")
    print("=" * 60)

if __name__ == "__main__":
    main()
