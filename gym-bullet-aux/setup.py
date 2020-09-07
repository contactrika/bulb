from setuptools import setup

setup(name='gym_bullet_aux',
      version='0.1',
      description='PyBullet envs with RGB camera observations that also include'
      'low-dimensional state in info for analyzing unsupervised learning algos',
      packages=['gym_bullet_aux'],
      author='contactrika',
      install_requires=[
        'gym', 'imageio', 'matplotlib', 'pybullet==2.8.1',  # was: pybullet==2.6.6
        #'gym_bullet_extensions@git+https://github.com/contactrika/bo-svae-dc.git#egg=gym_bullet_extensions-0.1&subdirectory=gym-bullet-extensions',
      ])
