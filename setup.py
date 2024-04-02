import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# Version
version_file = os.path.join(here, "urartu", "VERSION")
with open(version_file) as vf:
    __version__ = vf.read().strip()

# Requirements
with open(os.path.join(here, "requirements.txt"), "r") as f:
    requirements = f.read().splitlines()

# Package info
NAME = "urartu"
DESCRIPTION = "ML framework"
VERSION = __version__
REQUIRES_PYTHON = ">=3.10.0"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    python_requires=REQUIRES_PYTHON,
    install_requires=requirements,
    packages=find_packages("."),
    package_dir={"": "."},
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "urartu=urartu.__init__:main",
        ],
    },
)
