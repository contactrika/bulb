"""
A unified interface that reports high-dimensional observations
(images, point clouds) and also can report low-dimensional state (joint angles)
that can be used for further analysis.
"""

from abc import ABC, abstractmethod


class AuxEnv(ABC):
    """A constructor base that initializes basic flags and aggregators."""
    @abstractmethod
    def __init__(self,
                 max_episode_steps: int,
                 obs_resolution: bool = None,
                 obs_ptcloud: bool = None,
                 debug: bool = False,
                 visualize: bool = False):
        self._max_episode_steps = max_episode_steps
        self._obs_resolution = obs_resolution
        self._obs_ptcloud = obs_ptcloud
        self._debug = debug
        self._visualize = visualize
        # Aggregators.
        self._stepnum = 0
        self._episode_rwd = 0.0

    @property
    def max_episode_steps(self):
        """Maximum number of steps for this env. """
        return self._max_episode_steps

    @property
    def obs_resolution(self):
        """
        In case observations encode low-dim state (joint positions, velocities
        object positions and orientations, etc): obs_resolution is None.
        Otherwise it indicates pixel width and height of observations e.g. 64
        for 64x64 images, or the number of points in the point cloud.
        """
        return self._obs_resolution

    @property
    def obs_ptcloud(self):
        """A bool indication whether obs are be point clouds."""
        return self._obs_ptcloud

    @property
    def debug(self):
        """Debug flag. """
        return self._debug

    @property
    def visualize(self):
        """Visualization flag. """
        return self._visualize

    @property
    def stepnum(self):
        """Episode step counter. """
        return self._stepnum

    @property
    def episode_rwd(self):
        """Episode rewward aggregator. """
        return self._episode_rwd

    def reset_aggregators(self):
        """Reset step and episode reward aggregators."""
        self._stepnum = 0
        self._episode_rwd = 0

    def update_aggregators(self, rwd, done):
        """Update step and episode reward aggregators."""
        self._stepnum += 1
        self._episode_rwd += rwd
        info = {}
        if self._stepnum == self._max_episode_steps: done = True
        if done:
            info = {'r': float(self._episode_rwd), 'l': self._stepnum}
            if self._debug: print('tot_rwd {:.4f}'.format(self._episode_rwd))
        return done, info

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
