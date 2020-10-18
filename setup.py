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
            # they are not needed for the envs and are commented out to
            # make this repo lightweight
            #'torch', 'tensorboardX', 'moviepy', 'stable-baselines3'
      ],
      # Note: setup_requires does not install first, despite its name:
      # https://stackoverflow.com/questions/4996589/
      # in-setup-py-or-pip-requirements-file-how-to-control-order-of-installing-package
      # Hence, numpy should be installed separately with 'pip install numpy'
      # before using setup.py with 'pip install -e .'
      # This is because PyBullet does not have numpy as a dependency, but if
      # it find it during its own install, then point it will do processing of
      # point clouds using numpy, which will be faster.
      setup_requires=['numpy']
      )
