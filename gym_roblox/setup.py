from setuptools import setup, find_packages

setup(name='gym_roblox',
      description='Abstract classes for reinforcement learning problem using Roblox',
      version='0.0.1',
      install_requires=['gym>=0.2.3'],
      packages=find_packages(),
      include_package_data=True,
)
