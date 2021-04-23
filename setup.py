#!/usr/bin/env python
from os.path import join, dirname
from setuptools import setup, find_packages

with open(join(dirname(__file__), "README.md"), "r") as fh:
    long_description = fh.read()

# Setup package
setup(
    name="molecfit_wrapper",
    description="Molecfit Wrapper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    packages=find_packages(where=dirname(__file__)),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[],
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT",
    ],
)
