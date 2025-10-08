from setuptools import setup, find_packages

setup(
    name="mace-tools",
    version="0.1.0",
    description="Add on package for MACE containing functionality to model long range electrostatics",
    author="Harry Moore and Will Baldwin",
    entry_points={
        "console_scripts": [
            "mace-train = macetools.scripts.train:main",
        ]
    },
)
