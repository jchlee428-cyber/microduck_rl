"""
MicroDuck RL Lab - Environment Auto-Bootstrapper
Ensures scripts automatically execute inside the project's virtual environment (.venv),
preventing 'ModuleNotFoundError' even if the user forgets to activate .venv.
"""
import os
import sys
import subprocess

def ensure_venv():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(current_dir, ".venv", "Scripts", "python.exe")
    
    # Check if .venv python exists and is different from current python
    if os.path.exists(venv_python):
        cur_exe = os.path.normcase(os.path.abspath(sys.executable))
        target_exe = os.path.normcase(os.path.abspath(venv_python))
        
        if cur_exe != target_exe:
            # Re-launch with the virtualenv Python
            try:
                result = subprocess.run([target_exe] + sys.argv)
                sys.exit(result.returncode)
            except KeyboardInterrupt:
                sys.exit(0)

ensure_venv()
