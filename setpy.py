from setuptools import setup, find_packages

setup(
    name='FeGen',
    version='0.0.1',
    description='FeGen: A compiler frontend generator for fast DSL prototype.',
    author='chh',
    author_email='caohanghang23@mails.ucas.ac.cn',
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    install_requires=[
    ]
)