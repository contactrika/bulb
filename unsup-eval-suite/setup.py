from setuptools import setup

setup(name='unsup_eval_suite',
      version='0.1',
      description='Evaluation Suite for Unsupervised Learning',
      packages=['unsup_eval_suite'],
      author='contactrika',
      install_requires=['torch==1.4.0',
                        'numpy==1.18.1',  # to remove warnings
                        #'baselines==0.1.6',  # need to install from source
                        'pandas', 'gym', 'moviepy', 'tensorboardX'],
      )
