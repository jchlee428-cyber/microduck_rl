"""
MicroDuck RL Lab - TensorBoard Launcher
Automatically boots TensorBoard using the project virtual environment (.venv)
and opens the browser dashboard.
"""
import _bootstrap  # Automatically redirect to .venv python
import os
import sys
import time
import webbrowser
import subprocess

def main():
    print("=" * 60)
    print(" MicroDuck RL Lab: Launching TensorBoard Web Dashboard")
    print("=" * 60)
    
    url = "http://localhost:6006"
    print(f">> TensorBoard URL: {url}")
    print(">> Starting local server on port 6006...")
    print(">> Press Ctrl+C in this terminal to stop.")
    print("=" * 60)

    # Automatically open browser after 1.5 seconds
    import threading
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(url)
    threading.Thread(target=open_browser, daemon=True).start()

    # Run tensorboard module within venv
    try:
        subprocess.run([sys.executable, "-m", "tensorboard.main", "--logdir=logs", "--port=6006"])
    except KeyboardInterrupt:
        print("\n>> TensorBoard stopped.")

if __name__ == "__main__":
    main()
