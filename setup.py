"""
Setup of Deep LfD python codebase
Author: Michael Laskey
"""
from setuptools import setup

setup(name='fast_grasp_detect',
      version='0.1.dev0',
      description='Grasp Detection project code',
      author='Michael Laskey',
      author_email='laskeymd@berkeley.edu',
      package_dir = {'': 'src'},
      packages=['fast_grasp_detect', 'fast_grasp_detect.networks', 'fast_grasp_detect.labelers','fast_grasp_detect.data_aug','fast_grasp_detect.configs', 'fast_grasp_detect.core'],
     )
