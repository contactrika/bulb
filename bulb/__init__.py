import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

RESOLUTIONS = [None, 64, 128, 256, 512, 1024, 2048]
def resolution_suffix(resolution, obs_ptcloud):
    if resolution is None: return 'LD'
    suffix = 'PT' if obs_ptcloud else 'IM'
    return suffix+str(resolution)

def dbg_viz_suffix(dbg):
    return '' if dbg <= 0 else 'Dbg' if dbg <= 1 else 'Viz'

# Register standard gym envs and add Aux prefix.
# Note: not using 'Pusher', 'Striker', 'Thrower': they seem broken
gym_envs = ['CartPole', 'InvertedPendulum', 'InvertedDoublePendulum',
            'InvertedPendulumSwingup',
            'Hopper', 'Walker2D', 'HalfCheetah', 'Ant', 'Humanoid',
            'HumanoidFlagrun', 'HumanoidFlagrunHarder',
            'Reacher', 'Kuka']
v1_envs = ['CartPole']
for base_env_name in gym_envs:
    for dbg in [0, 1, 2]:
        for env_v in [0,1]:
            if env_v==0 and base_env_name=='CartPole': continue  # v1 only
            if env_v==1 and base_env_name not in v1_envs: continue
            for resolution in RESOLUTIONS:
                for obs_ptcloud in [False, True]:
                    if obs_ptcloud and resolution is None: continue
                    for random_colors in [False, True]:
                        if obs_ptcloud:
                            if random_colors: continue
                            if resolution is None: continue
                            if base_env_name != 'CartPole': continue
                        sfx0 = 'Clr' if random_colors else ''
                        sfx1 = resolution_suffix(resolution, obs_ptcloud)
                        sfx1 += dbg_viz_suffix(dbg)
                        env_id = base_env_name+sfx0+'BulletEnv'+sfx1+'-v'+str(env_v)
                        kwargs={'base_env_name':base_env_name,
                                'env_v':env_v,
                                'obs_resolution':resolution,
                                'obs_ptcloud':obs_ptcloud,
                                'random_colors':random_colors,
                                'debug': dbg==1, 'visualize': dbg==2}
                        register(id=env_id, entry_point='bulb.envs:AuxBulletEnv',
                                 kwargs=kwargs)
                        #print(env_id)

# Register rearrangement envs.
num_versions = 6*2  # 6 versions and their black-background variants
for robot in ['Reacher', 'Franka']:
    for variant in ['Ycb', 'OneYcb', 'Geom', 'OneGeom']:
        for resolution in RESOLUTIONS:
            for obs_ptcloud in [False, True]:
                if obs_ptcloud and resolution is None: continue
                for version in range(num_versions):
                    for dbg in [0, 1, 2]:
                        sfx = resolution_suffix(resolution, obs_ptcloud)
                        sfx += dbg_viz_suffix(dbg)
                        env_id = robot+'Rearrange'+variant+sfx+'-v'+str(version)
                        register(
                            id=env_id,
                            entry_point='bulb.envs:'+robot+'RearrangeEnv',
                            reward_threshold=1.0,
                            nondeterministic=True,
                            kwargs={'version': version,
                                    'variant': variant,
                                    'obs_resolution': resolution,
                                    'obs_ptcloud': obs_ptcloud,
                                    'rnd_init_pos': True,
                                    'statics_in_lowdim': False,
                                    'debug': dbg==1, 'visualize': dbg==2})
                        #print(env_id)

# Register BlockOnIncline envs: YcbOnIncline, GeomOnIncline, etc.
scale_dict = {'': 2.5, 'Md': 1.5, 'Sm': 1.0}
for variant in ['Ycb', 'Geom']:
    for scale_str, scale in scale_dict.items():
        for resolution in RESOLUTIONS:
            for obs_ptcloud in [False, True]:
                if obs_ptcloud and resolution is None: continue
                for version in range(6):
                    for dbg in [0, 1, 2]:
                        sfx = scale_str
                        sfx += resolution_suffix(resolution, obs_ptcloud)
                        sfx += dbg_viz_suffix(dbg)
                        env_id = variant+'OnIncline'+sfx+'-v'+str(version)
                        register(id=env_id,
                                 entry_point='bulb.envs:BlockOnInclineEnv',
                                 nondeterministic=True,
                                 kwargs={'version': version,
                                         'variant': variant,
                                         'obs_resolution': resolution,
                                         'obs_ptcloud': obs_ptcloud,
                                         'scale': scale,
                                         'randomize': True,
                                         'report_fric': False,
                                         'debug': dbg==1, 'visualize': dbg==2})
                        #print(env_id)
