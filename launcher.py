"""
MicroDuck RL Lab - Interactive Control Launcher
One-stop interactive console to view, evaluate, and monitor all trained robots.
"""
import _bootstrap  # Automatically redirect to .venv python
import os
import sys
import subprocess

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def print_banner():
    clear_screen()
    print("=" * 65)
    print("      🦆  MicroDuck RL Lab - Master Robot Control Center  🤖")
    print("=" * 65)
    print("  Welcome to the local Reinforcement Learning simulation hub.")
    print("  Select an action below to observe trained behaviors:")
    print("=" * 65)

def main_menu():
    while True:
        print_banner()
        print("  [ Featured Custom Robot ]")
        print("    D. 🦆  MicroDuck-v1     (Our Signature Duck Robot! Waddle-Walk)")
        print()
        print("  [ Standard 3D Benchmarks ]")
        print("    1. 🏋️  CartPole-v1      (Inverted Pendulum Balance)")
        print("    2. 🦿  Hopper-v5        (1-Leg Hopping Robot)")
        print("    3. 🏃  Walker2d-v5      (2-Leg Bipedal Walking Robot)")
        print("    4. 🐜  Ant-v5           (4-Leg Quadruped Crawling Robot)")
        print("    5. 🐆  HalfCheetah-v5   (Cheetah Forward Sprint Robot)")
        print("    6. 🦾  Humanoid-v5      (Full-Body Humanoid Robot - SOTA)")
        print()
        print("  [ Web Dashboard & Analytics ]")
        print("    W. 🌐  Launch Web Dashboard    (Streamlit Master GUI Center)")
        print("    7. 📊  View Learning Curves    (Open PNG image)")
        print("    8. 📈  Launch TensorBoard      (Web Dashboard)")
        print("    9. 🧪  Run Full Benchmark      (Evaluate All Robots)")
        print("    0. 🚪  Exit")
        print("=" * 65)

        choice = input("  Select an option [0-9, D, W]: ").strip().lower()

        if choice == "w":
            subprocess.run([sys.executable, "launch_dashboard.py"])
        elif choice == "d":
            print("\n>> Launching MicroDuck-v1 3D Viewer...")
            subprocess.run([sys.executable, "visualize_agent.py", "--env", "MicroDuck-v1", "--human", "--loop"])
        elif choice == "1":
            print("\n>> Launching CartPole-v1 3D Viewer...")
            subprocess.run([sys.executable, "visualize_agent.py", "--env", "CartPole-v1", "--human", "--loop"])
        elif choice == "2":
            print("\n>> Launching Hopper-v5 3D Viewer...")
            subprocess.run([sys.executable, "visualize_agent.py", "--env", "Hopper-v5", "--human", "--loop"])
        elif choice == "3":
            print("\n>> Launching Walker2d-v5 3D Viewer...")
            subprocess.run([sys.executable, "visualize_agent.py", "--env", "Walker2d-v5", "--human", "--loop"])
        elif choice == "4":
            print("\n>> Launching Ant-v5 3D Viewer...")
            subprocess.run([sys.executable, "visualize_agent.py", "--env", "Ant-v5", "--human", "--loop"])
        elif choice == "5":
            print("\n>> Launching HalfCheetah-v5 3D Viewer...")
            subprocess.run([sys.executable, "visualize_agent.py", "--env", "HalfCheetah-v5", "--human", "--loop"])
        elif choice == "6":
            print("\n>> Launching Humanoid-v5 3D Viewer...")
            subprocess.run([sys.executable, "visualize_agent.py", "--env", "Humanoid-v5", "--human", "--loop"])
        elif choice == "7":
            img_path = os.path.abspath("logs/learning_curves.png")
            if os.path.exists(img_path):
                print(f"\n>> Opening {img_path}...")
                os.system(f'start "" "{img_path}"')
            else:
                print("\n>> Generating plot first...")
                subprocess.run([sys.executable, "plot_results.py"])
                os.system(f'start "" "{img_path}"')
            input("\nPress Enter to return to menu...")
        elif choice == "8":
            subprocess.run([sys.executable, "launch_tensorboard.py"])
        elif choice == "9":
            print("\n>> Running benchmark evaluation on MicroDuck, CartPole, and all MuJoCo robots...")
            subprocess.run([sys.executable, "evaluate_mujoco.py", "--env", "MicroDuck-v1", "--episodes", "3", "--no-compare"])
            subprocess.run([sys.executable, "evaluate_cartpole.py", "--episodes", "3", "--no-compare"])
            for env_name in ["Hopper-v5", "Walker2d-v5", "Ant-v5", "HalfCheetah-v5", "Humanoid-v5"]:
                subprocess.run([sys.executable, "evaluate_mujoco.py", "--env", env_name, "--episodes", "3", "--no-compare"])
            input("\nBenchmark completed. Press Enter to return to menu...")
        elif choice == "0":
            print("\nThank you for using MicroDuck RL Lab! Bye 🦆")
            break
        else:
            print("\nInvalid choice. Please enter a number from 0 to 9.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nExited.")
