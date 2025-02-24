import os
from typing import List

from setuptools import find_packages, setup

pwd = os.path.dirname(__file__)


def readme():
    with open(os.path.join(pwd, "README.md")) as f:
        content = f.read()
    return content


def get_version():
    with open(os.path.join(pwd, "version.txt"), encoding="utf-8") as f:
        content = f.read()
    return content


def get_requires() -> List[str]:
    with open(os.path.join("requirements", "runtime.txt"), encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


extra_require = {
    "torch": ["torch>=2.1.0"],
    "torch-npu": ["torch==2.1.0", "torch-npu==2.1.0.post3", "numpy==1.26.4", "scipy", "decorator"],
}

setup(
    name="InternEvo",
    version=get_version(),
    description="Lightweight training framework for LLM",
    author="InternEvo team",
    license="Apache 2.0 License",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests"]),
    install_requires=get_requires(),
    extras_require=extra_require,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
    ],
)
