"""
Windows Unicode Path Compatibility Patch for MuJoCo in Gymnasium
-----------------------------------------------------------------
In Windows environments where the file path contains non-ASCII characters
(such as Korean '바탕 화면'), MuJoCo's C library function `from_xml_path`
fails to open XML files (ParseXML: Error opening file).

This patch intercepts Gymnasium's `MujocoEnv._initialize_simulation`
to read the model file using Python's UTF-8 file loader and pass it
via `mujoco.MjModel.from_xml_string`, ensuring 100% compatibility.
"""
import mujoco
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

_original_initialize_simulation = MujocoEnv._initialize_simulation

def _patched_initialize_simulation(self):
    try:
        # First attempt default behavior
        return _original_initialize_simulation(self)
    except (ValueError, OSError):
        # Fallback to UTF-8 XML string parsing
        with open(self.fullpath, "r", encoding="utf-8") as f:
            xml_string = f.read()
        model = mujoco.MjModel.from_xml_string(xml_string)
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

def apply_mujoco_windows_patch():
    MujocoEnv._initialize_simulation = _patched_initialize_simulation

    # Wrap mujoco.mjv_moveCamera for Gymnasium WindowViewer mouse interaction compatibility
    _orig_moveCamera = mujoco.mjv_moveCamera
    def _compat_moveCamera(*args, **kwargs):
        if len(args) == 6:
            # Drop 5th arg (scn) -> (m, action, reldx, reldy, cam)
            return _orig_moveCamera(args[0], args[1], args[2], args[3], args[5])
        return _orig_moveCamera(*args, **kwargs)
    mujoco.mjv_moveCamera = _compat_moveCamera

# Auto-apply on import
apply_mujoco_windows_patch()
