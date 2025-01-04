from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="FeGen",
    version="0.0.1",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10.14",
)
