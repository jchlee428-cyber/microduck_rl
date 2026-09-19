"""
MicroDuck-v1: Custom Gymnasium MuJoCo Environment for MicroDuck Bipedal Robot
=============================================================================
A 2-legged waddling duck robot with 6 motorized joints (hip, knee, ankle on each leg)
and flat webbed feet designed for learning stable forward locomotion (Waddle-Walk).
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
import mujoco_compat  # Windows Unicode path patch & camera fix

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 2.2,
    "elevation": -14.0,
    "azimuth": 135.0,
}

class MicroDuckEnv(MujocoEnv, utils.EzPickle):
    """
    MicroDuck Gymnasium MuJoCo Environment.
    
    Observation Space (23 dimensions):
      - qpos[2]: Torso Z height (1)
      - qpos[3:7]: Torso orientation quaternion [qw, qx, qy, qz] (4)
      - qpos[7:13]: 6 Joint angles [L_hip, L_knee, L_ankle, R_hip, R_knee, R_ankle] (6)
      - qvel[0:3]: Torso linear velocity [vx, vy, vz] (3)
      - qvel[3:6]: Torso angular velocity [wx, wy, wz] (3)
      - qvel[6:12]: 6 Joint angular velocities (6)
      
    Action Space (6 dimensions continuous):
      - 6 Joint motor torques in [-1.0, 1.0]
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 4,
        default_camera_config: dict = None,
        forward_reward_weight: float = 1.5,
        ctrl_cost_weight: float = 0.005,
        healthy_reward: float = 1.0,
        lateral_drift_weight: float = 0.5,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple = (0.22, 0.65),
        reset_noise_scale: float = 0.02,
        **kwargs,
    ):
        if xml_file is None:
            xml_file = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "microduck.xml")
            )

        if default_camera_config is None:
            default_camera_config = DEFAULT_CAMERA_CONFIG

        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            lateral_drift_weight,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._lateral_drift_weight = lateral_drift_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        # MuJoCo initialization
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # 23-dim observation space: (13 qpos - 2 [x,y]) + 12 qvel = 23
        obs_size = (self.data.qpos.size - 2) + self.data.qvel.size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    @property
    def is_healthy(self) -> bool:
        """Duck is healthy if upright and torso height is within comfortable walking range."""
        z = self.data.qpos[2]
        min_z, max_z = self._healthy_z_range
        is_upright = (min_z <= z <= max_z)
        
        # Check roll & pitch from quaternion to ensure duck hasn't toppled over
        # qpos[3:7] = [qw, qx, qy, qz]
        qw, qx, qy, qz = self.data.qpos[3:7]
        # Up-vector Z component in world frame = 1 - 2*(qx^2 + qy^2)
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        not_flipped = up_z > 0.3  # Torso is facing reasonably upwards
        
        return bool(is_upright and not_flipped)

    def _get_obs(self):
        """Construct the 23-dimensional normalized observation vector."""
        # Exclude x, y coordinates to preserve translational invariance
        qpos = self.data.qpos[2:].copy()
        qvel = np.clip(self.data.qvel.copy(), -10.0, 10.0)
        return np.concatenate([qpos, qvel]).astype(np.float64)

    def step(self, action):
        x_before = self.data.qpos[0]
        y_before = self.data.qpos[1]

        # Apply continuous joint torque simulation
        self.do_simulation(action, self.frame_skip)

        x_after = self.data.qpos[0]
        y_after = self.data.qpos[1]

        dt = self.dt
        vx = (x_after - x_before) / dt
        vy = (y_after - y_before) / dt

        # Reward components
        forward_reward = self._forward_reward_weight * vx
        healthy_reward = self._healthy_reward if self.is_healthy else 0.0
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        lateral_cost = self._lateral_drift_weight * abs(vy)

        reward = forward_reward + healthy_reward - ctrl_cost - lateral_cost

        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        truncated = False
        obs = self._get_obs()

        info = {
            "x_position": x_after,
            "x_velocity": vx,
            "y_velocity": vy,
            "forward_reward": forward_reward,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
            "is_healthy": self.is_healthy,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def reset_model(self):
        """Reset the duck to natural standing posture with slight exploration noise."""
        noise_scale = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=-noise_scale, high=noise_scale, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-noise_scale, high=noise_scale, size=self.model.nv
        )
        
        # Ensure duck starts cleanly at nominal standing height (approx 0.38m)
        qpos[0] = 0.0  # x
        qpos[1] = 0.0  # y
        qpos[2] = 0.38  # z standing height
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion (perfectly upright)

        self.set_state(qpos, qvel)
        return self._get_obs()


# Register environment with Gymnasium
gym.register(
    id="MicroDuck-v1",
    entry_point="microduck_env:MicroDuckEnv",
    max_episode_steps=1000,
)
