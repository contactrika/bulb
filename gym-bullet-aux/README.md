## gym-bullet-aux

This package contains pyBullet benchmark environments that are extended to report both high- and low-dimensional state and the new *RearrangeGeom* and *RearrangeYCB* environments.

![pyBullet benchmarks](gym_bullet_aux/envs/data/img/pyBullet_benchmarks.png)
![Rearrange envs](gym_bullet_aux/envs/data/img/Rearrange_and_YCB.png)

### Installation

```
virtualenv --no-site-packages -p /usr/bin/python3.6 AUX_ENV
source AUX_ENV/bin/activate
pip install numpy
pip install -e "git+https://github.com/contactrika/bo-svae-dc.git#egg=gym_bullet_extensions-0.1&subdirectory=gym-bullet-extensions"
cd gym-bullet-aux
pip install -e .
```

### Usage Examples
```
python -m gym_bullet_aux.aux_env_demo --env_name=ReacherRearrangeGeom64-v0 --num_episodes=10 --debug --viz
python -m gym_bullet_aux.aux_env_demo --env_name=BlockOnInclineYcb-v0 --debug --viz
python -m gym_bullet_aux.aux_env_demo --env_name=BlockOnInclineYcb-v3 --debug --viz
```

### Running  Benchmarks

The following behcmark environments are available:
```AuxInvertedPendulumBulletEnv-v0, AuxInvertedDoublePendulumBulletEnv-v0, AuxInvertedPendulumSwingupBulletEnv-v0, AuxHopperBulletEnv-v0, AuxWalker2DBulletEnv-v0, AuxHalfCheetahBulletEnv-v0, AuxAntBulletEnv-v0, AuxKukaBulletEnv-v0```

To get low-dimensional simulator state in observations (instead of RGB observations) add ```LD``` suffix, e.g. ```AuxInvertedPendulumLDBulletEnv-v0```.

To get a version of domain that would be visualized add ```Viz``` suffix: ```AuxInvertedPendulumBulletEnvViz-v0```

The above have continuous action space, ```AuxCartPoleBulletEnv-v1``` is available with discrete action space.

```
python -m gym_bullet_aux.aux_env_demo --env_name=AuxAntBulletEnv-v1 --debug --viz
```

<br />
<hr />

<sub>**A note about this implementation:** The code in this package is for basic academic experiments.
It favors simplicity over performance and does not try to follow any particular style guidelines. It would be organized differently if we aimed for
reliability/deployment in an industrial setting.</sub>
