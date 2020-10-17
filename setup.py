from setuptools import setup

setup(name='bulb',
      version='0.1',
      description='PyBullet envs with RGB of point cloud observations'
                  'that also report low-dimensional state for analysis',
      packages=['bulb'],
      author='contactrika',
      install_requires=[
            'gym', 'imageio', 'matplotlib',
            'pybullet >=2.6.4',  # note: tested until <=2.8.1
            # Note: additional dependencies below are for rl_demo.py
            # they are not needed for envs and are commented out to
            # make this repo lightweight
            #'torch', 'tensorboardX', 'moviepy', 'stable-baselines3'
      ])
