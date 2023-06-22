from pathlib import Path
from setuptools import find_namespace_packages , setup

BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR,'requirements.txt') ,'r') as file:
    required_packages=[ln.strip() for ln in file.readlines()]


setup(
    name="duplicate questions",
    version=0.1,
    description="Classify duplicate questions.",
    author="Vishal Gaikwad",
    author_email="gaikwadvishal114@gmail.com",
    url="",
    python_requires=">=3.8",
    install_requires=[required_packages],
)