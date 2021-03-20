from setuptools import setup, find_packages

setup(name='maddpg',
      version='0.0.1',
      description='Pytorch implementation of Multi-Agent Deep Deterministic Policy Gradient',
      url='https://github.com/cying9/maddpg',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
