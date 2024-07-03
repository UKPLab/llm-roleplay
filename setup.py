import os
from pathlib import Path

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

version_file = os.path.join(here, "llm_roleplay", "VERSION")
with open(version_file) as vf:
    __version__ = vf.read().strip()

# Requirements
with open(os.path.join(here, "requirements.txt"), "r") as f:
    requirements = f.read().splitlines()

# Package info
NAME = "llm-roleplay"
DESCRIPTION = "LLM Roleplay: Simulating Human-Chatbot Interaction"
VERSION = __version__
REQUIRES_PYTHON = ">=3.9.0"
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=REQUIRES_PYTHON,
    install_requires=requirements,
    packages=find_packages("."),
    package_dir={"": "."},
    zip_safe=False,
)
