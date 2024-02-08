#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="spp_paul",
    version="0.0.0",
    description="code to train and run SPP estimator",
    author="Marvin Tammen",
    author_email="marvin.tammen@uol.de",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="https://github.com/PyTorchLightning/pytorch-lightning-conference-seed",
    install_requires=["pytorch-lightning"],
    packages=find_packages(),
)
