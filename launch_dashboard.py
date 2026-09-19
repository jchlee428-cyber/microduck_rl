"""
MicroDuck RL Lab - Streamlit Web Dashboard Launcher
===================================================
Launches the Streamlit web control center and opens it in the browser.
"""
import _bootstrap
import os
import sys
import time
import webbrowser
import subprocess

def launch():
    print("=" * 65)
    print("      🌐  MicroDuck RL Lab - Launching Web Dashboard  🦆")
    print("=" * 65)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("[INFO] Installing streamlit in .venv...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])

    port = 8501
    url = f"http://127.0.0.1:{port}"
    print(f"[INFO] Starting Streamlit server on {url} ...")
    print("[INFO] Dashboard includes all 7 robots, GIF showcase, telemetry & analytics.")
    print("=" * 65 + "\n")

    # Start streamlit process
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "false",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
        "--browser.serverAddress",
        "127.0.0.1"
    ]
    
    # Open browser automatically after a short delay
    def _open_browser():
        time.sleep(2.0)
        webbrowser.open(url)

    import threading
    t = threading.Thread(target=_open_browser)
    t.daemon = True
    t.start()

    subprocess.run(cmd)

if __name__ == "__main__":
    try:
        launch()
    except KeyboardInterrupt:
        print("\n[INFO] Dashboard server stopped.")
