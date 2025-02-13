import os
from typing import List
from setuptools import setup, find_packages
from setuptools.command.install import install

pwd = os.path.dirname(__file__)

def readme():
    with open(os.path.join(pwd, 'README.md')) as f:
        content = f.read()
    return content

def get_version():
    with open(os.path.join(pwd, 'version.txt'), 'r') as f:
        content = f.read()
    return content

def get_requires() -> List[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines

extra_require = {
    "torch": ["torch>=1.13.1"],
    "torch-npu": ["torch==2.1.0", "torch-npu==2.1.0.post3"],
    "test": ["pre-commit", "pylint", "pytest"],
}

def get_console_scripts() -> List[str]:
    console_scripts = ["internevo-cli = internevo.cli:main"]
    return console_scripts

setup(
    name='InternEvo',
    version=get_version(),
    description='an open-sourced lightweight training framework aims to support model pre-training without the need for extensive dependencies',
    long_description=readme(),
    long_description_content_type='text/markdown',
    package_dir={"": "internevo"},
    packages=find_packages("internevo"),
    install_requires=get_requires(),
    extras_require=extra_require,
    entry_points={"console_scripts": get_console_scripts()},
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ],
)
