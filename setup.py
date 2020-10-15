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
            'torch', 'tensorboardX', 'moviepy',
            #'gym_bullet_extensions@git+https://github.com/contactrika/bo-svae-dc.git#egg=gym_bullet_extensions-0.1&subdirectory=gym-bullet-extensions',
      ])
