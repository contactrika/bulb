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
python -m gym_bullet_aux.aux_env_demo --env_name=ReacherRearrangeGeom64-v0 --num_episodes=10 --viz
python -m gym_bullet_aux.aux_env_demo --env_name=ReacherRearrangeYcb64-v0 --num_episodes=10 --debug
```


<br />
<hr />

<sub>**A note about this implementation:** The code in this package is for basic academic experiments.
It favors simplicity over performance and does not try to follow any particular style guidelines. It would be organized differently if we aimed for
reliability/deployment in an industrial setting.</sub>
