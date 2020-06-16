## Benchmarking Unsupervised Learning with pyBullet

This repository contains code for the Evaluation Suite presented in:
*"Analytic Manifold Learning: Unifying and Evaluating Representations for Continuous Control"*
```
@article{antonova2020analytic,
  title={Analytic Manifold Learning: Unifying and Evaluating Representations for Continuous Control},
  author={Rika Antonova and Maksim Maydanskiy and Danica Kragic and Sam Devlin and Katja Hofmann},
  journal={arXiv preprint arXiv:TODO},
  year={2020}
}
```

![pyBullet benchmarks](unsup-eval-suite/plots/ant_all_err.png)
![pyBullet benchmarks](gym-bullet-aux/gym_bullet_aux/envs/data/img/pyBullet_benchmarks.png)
![Rearrange envs](gym-bullet-aux/gym_bullet_aux/envs/data/img/Rearrange_and_YCB.png)

There are two packages in this repository:

[gym-bullet-aux](gym-bullet-aux) contains pyBullet benchmark environments that are extended to report both high- and low-dimensional state and the new *RearrangeGeom* and *RearrangeYCB* environments

[unsup-eval-suite](unsup-eval-suite) contains unsupervised learners (VAE, SVAE, PRED, DSA) and functionality for measuring alignment between the learned latent state and the true low-dimensional (simulator) state; also contains an interface with a PPO RL learner


## Installation

See READMEs in [gym-bullet-aux](gym-bullet-aux) and [unsup-eval-suite](unsup-eval-suite)


## Example Usage

See READMEs in [gym-bullet-aux](gym-bullet-aux) and [unsup-eval-suite](unsup-eval-suite)
