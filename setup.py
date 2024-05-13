import os
import re
import sys
import subprocess
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

def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines() if 'torch-scatter' not in r and not r.startswith('-f ')]

setup(
    name='InternEvo',
    version=get_version(),
    description='an open-sourced lightweight training framework aims to support model pre-training without the need for extensive dependencies',
    long_description=readme(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        fetch_requirements('requirements/runtime.txt'),
        'rotary_emb',
        'xentropy',
    ],

    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ],
)
