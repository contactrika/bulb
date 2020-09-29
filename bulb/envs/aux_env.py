"""
A unified interface that reports high-dimensional observations
(images, point clouds) and also adds low-dimensional state (joint angles)
into info['low_dim_state'] that can be used for further analysis.
"""

from abc import ABC, abstractmethod


class AuxEnv(ABC):
    @property
    @abstractmethod
    def low_dim_state_space(self):
        """
        Low dimensional observation space. Similar to env.observation_space
        property (e.g. gym.spaces.Box).
        """
        pass

    @property
    @abstractmethod
    def low_dim_state_names(self):
        """
        Short names for each dimension/entry in the low-dimensional state.
        """
        pass

    @property
    @abstractmethod
    def max_episode_steps(self):
        """
        Maximum number of steps for this env.
        """
        pass

    @abstractmethod
    def compute_low_dim_state(self):
        """
        Compute the current low-dimensional state of the simulation.
        This might include robot joint angles and velocities, object positions
        and orientations, etc.
        """
        pass

    @abstractmethod
    def override_state(self, ld_state):
        """
        Overrides the current simulation state to be consistent with the given
        ld_state, which contains low-dimensional state information, e.g. robot
        joint angles and velocities, object positions and orientations, etc.
        """
        pass

    @abstractmethod
    def render_obs(self, override_resolution=None, debug_out_dir=None):
        """
        Render current observations as an RGB pixel image or a point cloud.
        """
        pass
