from setuptools import setup, find_packages

# Load requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()
    
setup(
    name='cs_returns',
    version='0.1.0',
    description='End to end pipeline to predict orders with highest probability of being returned',
    author='Shaun Shibu',
    author_email='shaunchackoofficial@gmail.com',
    packages=find_packages(exclude=('tests',)),
    install_requires=required,
)
