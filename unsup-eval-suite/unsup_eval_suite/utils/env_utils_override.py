"""
Vectorized env with override_state() to override simulation state.
"""
import numpy as np
import multiprocessing as mp

from .baselines_common.vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars
from .baselines_common.vec_env.dummy_vec_env import DummyVecEnv
from .baselines_common.vec_env.shmem_vec_env import ShmemVecEnv, _NP_TO_CT
from .baselines_common.vec_env.util import obs_space_info, obs_to_dict

from .env_utils import make_env, PyTorchVecEnv

import gym_bullet_aux  # to register envs


def make_vec_envs_override(env_name, env_args, seed, num_envs, device):
    if env_name.startswith(('Reacher','Franka','BlockOnIncline')):
        num_envs_types = 6  # we have 6 types of rearrange envs
        num_envs_for_mod = 6 if env_name.startswith('BlockOnIncline') else 4
        any_env_skin = env_args.any_env_skin
        envs = []
        for i in range(num_envs):
            v = i%num_envs_for_mod if any_env_skin else env_args.env_skin_id
            if any_env_skin>1: v += (any_env_skin-1)*num_envs_types
            full_env_name = env_name+'-v'+str(v)
            envs.append(make_env(full_env_name, seed+i, i))
    else:
        envs = [make_env(env_name, seed, i) for i in range(num_envs)]
    if len(envs) > 1:
        try:
            envs = ShmemVecEnv(envs, context='fork')
        except:
            envs = ShmemVecEnv(envs)  # for older versions remove context='fork'
    else:
        envs = DummyVecEnv(envs)
    envs = PyTorchVecEnv(envs, device)
    return envs


class ShmemOverrideVecEnv(ShmemVecEnv):
    def __init__(self, env_fns, spaces=None, context='spawn'):
        """
        ShmemVecEnv was not easily extendable, so needed to copy a large part
        of its constructor from baselines_common.vec_env.shmem_vec_env
        """
        ctx = mp.get_context(context)
        if spaces:
            observation_space, action_space = spaces
        else:
            dummy = env_fns[0]()
            observation_space, action_space = dummy.observation_space, dummy.action_space
            dummy.close()
            del dummy
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space)
        self.obs_bufs = [
            {k: ctx.Array(_NP_TO_CT[self.obs_dtypes[k].type], int(np.prod(self.obs_shapes[k]))) for k in self.obs_keys}
            for _ in env_fns]
        self.parent_pipes = []
        self.procs = []
        with clear_mpi_env_vars():
            for env_fn, obs_buf in zip(env_fns, self.obs_bufs):
                wrapped_fn = CloudpickleWrapper(env_fn)
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(target=_subproc_worker,
                            args=(child_pipe, parent_pipe, wrapped_fn, obs_buf, self.obs_shapes, self.obs_dtypes, self.obs_keys))
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()
        self.waiting_step = False
        self.viewer = None


def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_bufs, obs_shapes, obs_dtypes, keys):
    """
    _subproc_worker was not easliy extendable, so needed to copy a large part
    from baselines_common.vec_env.shmem_vec_env to add override_state
    """
    def _write_obs(maybe_dict_obs):
        flatdict = obs_to_dict(maybe_dict_obs)
        for k in keys:
            dst = obs_bufs[k].get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])  # pylint: disable=W0212
            np.copyto(dst_np, flatdict[k])

    env = env_fn_wrapper.x()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                pipe.send(_write_obs(env.reset()))
            elif cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                pipe.send((_write_obs(obs), reward, done, info))
            elif cmd == 'render':
                pipe.send(env.render(mode='rgb_array'))
            elif cmd == 'override_state':
                env.override_state(data)
                pipe.send(None)
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('ShmemVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class PyTorchOverrideVecEnv(PyTorchVecEnv):
    def __init__(self, venv, device, squeeze_actions=False):
        super(PyTorchOverrideVecEnv, self).__init__(venv, device, squeeze_actions)

    def override_state(self, states):
        # Note: users have to ensure that the underlying env supports override
        if self.venv.waiting_step: self.venv.step_wait()
        assert len(states) == len(self.venv.parent_pipes)
        for pipe, state in zip(self.venv.parent_pipes, states):
            pipe.send(('override_state', state))
        res = [pipe.recv() for pipe in self.venv.parent_pipes]
