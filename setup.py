# -*- coding: utf-8 -*-
""" 

Created on 06/07/18

Author : Carlos Eduardo Barbosa

"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bsf",
    version="0.0.1",
    author="Carlos Eduardo Barbosa",
    author_email="kadu.barbosa@gmail.com",
    description="Bayesian Spectral Fitting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/cebarbosa/bsf",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)