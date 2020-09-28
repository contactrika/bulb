import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Register standard gym envs and add Aux prefix.
# Note: not using 'Pusher', 'Striker', 'Thrower': they seem broken
gym_envs = ['CartPole', 'InvertedPendulum', 'InvertedDoublePendulum',
            'InvertedPendulumSwingup',
            'Hopper', 'Walker2D', 'HalfCheetah', 'Ant', 'Humanoid',
            'HumanoidFlagrun', 'HumanoidFlagrunHarder',
            'Reacher', 'Kuka']
v1_envs = ['CartPole']
for base_env_name in gym_envs:
    for debug_level in [0, 1, 2]:
        for env_v in [0,1]:
            if env_v==0 and base_env_name=='CartPole': continue  # v1 only
            if env_v==1 and base_env_name not in v1_envs: continue
            for resolution in [None, 64, 128, 256, 512, 1024, 2048]:
                for obs_ptcloud in [False, True]:
                    for random_colors in [False, True]:
                        if obs_ptcloud:
                            if random_colors: continue
                            if resolution is None: continue
                            if base_env_name != 'CartPole': continue
                        sfx0 = ''
                        if random_colors: sfx0 += 'Clr'
                        if resolution is None: sfx0 += 'LD'
                        if obs_ptcloud: sfx0 += 'PT'
                        if resolution not in (None, 64): sfx0 += str(resolution)
                        sfx1 = '' if debug_level<=0 else 'Debug' if debug_level<=1 else 'Viz'
                        env_id = 'Aux'+base_env_name+sfx0+'BulletEnv'+sfx1+'-v'+str(env_v)
                        kwargs={'base_env_name':base_env_name, 'env_v':env_v,
                                'obs_resolution':resolution, 'obs_ptcloud':obs_ptcloud,
                                'random_colors':random_colors,
                                'visualize':(debug_level>=2), 'debug_level':debug_level}
                        register(id=env_id, entry_point='gym_bullet_aux.envs:AuxBulletEnv',
                                 kwargs=kwargs)
                        #print(env_id)

# Register rearrangement envs.
num_versions = 6*2  # 6 versions and their black-background variants
max_episode_steps = 50  # pass this explicitly to env instead of wrapping
for robot in ['Ureacher', 'Reacher', 'Franka']:
    for variant in ['Ycb', 'OneYcb', 'Geom', 'OneGeom']:
        for rnd_init_pos in [False, True]:
            for resolution in [None, 64, 128, 256, 512, 1024, 2048]:
                for obs_ptcloud in [False, True]:
                    if obs_ptcloud and resolution is None: continue
                    for version in range(num_versions):
                        for debug_level in [0, 1, 2]:
                            suffix = '' if rnd_init_pos else 'Nornd'
                            if obs_ptcloud: suffix += 'PT'
                            suffix += 'LD' if resolution is None else str(resolution)
                            suffix += '' if debug_level<=0 else 'Debug' if debug_level<=1 else 'Viz'
                            env_id = robot+'Rearrange'+variant+suffix+'-v'+str(version)
                            register(
                                id=env_id,
                                entry_point='gym_bullet_aux.envs:'+robot+'RearrangeEnv',
                                reward_threshold=1.0,
                                nondeterministic=True,
                                kwargs={'version':version, 'variant':variant,
                                        'max_episode_steps':max_episode_steps,
                                        'obs_resolution':resolution,
                                        'obs_ptcloud':obs_ptcloud,
                                        'rnd_init_pos':rnd_init_pos,
                                        'statics_in_lowdim':False,
                                        'visualize':(debug_level>=2),
                                        'debug_level':debug_level})
                            #print(env_id)

# Register BlockOnIncline
scale_dict = {'': 2.5, 'Md': 1.5, 'Sm': 1.0}
max_episode_steps = 50  # pass this explicitly to env instead of wrapping
for variant in ['Ycb', 'Geom', 'YcbFric', 'GeomFric',
                'YcbLD', 'GeomLD', 'YcbFricLD', 'GeomFricLD',
                'YcbNorndLD', 'GeomNorndLD']:
    obs_resolution=64; randomize=True; report_fric=False; variant_arg=variant
    if variant_arg.endswith('LD'):
        obs_resolution = None; variant_arg = variant[:-2]
    if variant_arg.endswith('Fric'):
        report_fric = True; variant_arg = variant_arg[:-4]
    if variant_arg.endswith('Nornd'):
        randomize = False; variant_arg = variant_arg[:-5]
    for version in range(6):
        for scale_str, scale in scale_dict.items():
                for debug_level in [0, 1, 2]:
                    suffix = scale_str
                    suffix += '' if debug_level<=0 else 'Debug' if debug_level<=1 else 'Viz'
                    rid = 'BlockOnIncline'+variant+suffix+'-v'+str(version)
                    register(id=rid, entry_point='gym_bullet_aux.envs:BlockOnInclineEnv',
                             nondeterministic=True,
                             kwargs={'version': version, 'variant': variant_arg,
                                     'max_episode_steps':max_episode_steps,
                                     'scale': scale, 'randomize': randomize,
                                     'report_fric': report_fric,
                                     'obs_resolution': obs_resolution,
                                     'obs_ptcloud':False,
                                     'visualize':(debug_level>=2),
                                     'debug_level':debug_level})


# Register BlockOnIncline with poit clouds.
scale_dict = {'': 2.5, 'Md': 1.5, 'Sm': 1.0}
for variant in ['Ycb', 'Geom']:
    for obs_resolution in [64, 128, 256, 512, 1024, 2048]:
        for version in range(6):
            for scale_str, scale in scale_dict.items():
                    for debug_level in [0, 1, 2]:
                        sfx1 = '' if debug_level<=0 else 'Debug' if debug_level<=1 else 'Viz'
                        rid = 'BlockOnIncline'+variant+scale_str+'PT'+str(obs_resolution)+sfx1+'-v'+str(version)
                        register(id=rid, entry_point='gym_bullet_aux.envs:BlockOnInclineEnv',
                                 nondeterministic=True,
                                 kwargs={'version': version, 'variant': variant,
                                         'scale': scale, 'randomize': False,
                                         'report_fric': False,
                                         'obs_resolution': obs_resolution,
                                         'obs_ptcloud':True,
                                         'visualize':(debug_level>=2),
                                         'debug_level':debug_level})
