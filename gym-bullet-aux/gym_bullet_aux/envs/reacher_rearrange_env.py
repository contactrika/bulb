"""
Rearrangement env for a simple 2-link Reacher.
"""

from .reacher_sim import ReacherBulletSimulation
from .rearrange_env import RearrangeEnv


class ReacherRearrangeEnv(RearrangeEnv):
    def __init__(self, version, max_episode_len,
                 obs_resolution=64, obs_ptcloud=False, variant='Ycb',
                 rnd_init_pos=False, statics_in_lowdim=False,
                 visualize=False, debug_level=0):
        self.robot = ReacherBulletSimulation(
            robot_desc_file='reacher.xml', gui=visualize, camera_distance=0.40) #0.60)
        # Note: RearrangeEnv expects that we created self.robot already.
        super(ReacherRearrangeEnv, self).__init__(
            version=version, max_episode_len=max_episode_len,
            obs_resolution=obs_resolution, obs_ptcloud=obs_ptcloud,
            variant=variant,
            rnd_init_pos=rnd_init_pos, statics_in_lowdim=statics_in_lowdim,
            debug_level=debug_level)
