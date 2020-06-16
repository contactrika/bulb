## unsup-eval-suite

This package contains unsupervised learners (VAE, SVAE, PRED, DSA) and functionality for measuring alignment between the learned latent state and the true low-dimensional (simulator) state; also contains an interface with a PPO RL learner.

![pyBullet benchmarks](plots/ant_all_err.png =210x)
![Rearrange envs](plots/RearrangeReacher_ori_results.jpg =500x)

### Install

```
virtualenv --no-site-packages -p /usr/bin/python3.6 AUX_ENV
source AUX_ENV/bin/activate
cd unsup-eval-suite
pip install -e .
```

### Usage Examples

Coming soon


<br />
<hr />

<sub>**A note about this implementation:** The code in this package is for basic academic experiments.
It favors simplicity over performance and does not try to follow any particular style guidelines. It would be organized differently if we aimed for
reliability/deployment in an industrial setting.</sub>
