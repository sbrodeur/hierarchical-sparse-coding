#!/usr/bin/env python

import os
from setuptools import setup

setup(
    name = "Hierarchical sparse coding (HSC)",
    version = "1.0.0",
    author = "Simon Brodeur",
    author_email = "Simon.Brodeur@USherbrooke.ca",
    description = ("Hierarchical sparse coding implementation in Python, and related theoretical analysis and datasets."),
    license = "BSD 3-Clause License",
    keywords = "signal processing, unsupervised learning, artificial intelligence, information compression",
    url = "https://github.com/sbrodeur/hierarchical-sparse-coding",
    packages=['hsc', 'tests'],
    setup_requires=['setuptools-markdown'],
    install_requires=[
        "setuptools-markdown",
        "numpy",
        "scipy",
        "matplotlib",
    ],
    long_description_markdown_filename='README.md',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "License :: OSI Approved :: BSD License",
    ],
)
