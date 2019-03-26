#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup


NAME = "Cellua"
AUTHOR = "HactarCE"

try:
    with open('VERSION') as f:
        VERSION = f.read().strip()
except:
    VERSION = None

setup(
    name=NAME,
    version=VERSION,
    description="Multidimensional cellular automaton simulator",
    long_description="Cellua is a fast, flexible, and feature-rich multidimensional cellular automaton simulator.",
    author=AUTHOR,
    python_requires='>=3.7',
    url='https://github.com/{}/{}'.format(AUTHOR, NAME),
    install_requires=[
        'lupa>=1.8',
        'numpy>=1.16',
    ],
    extras_require={
        'dev': [
            'hypothesis>=4.12',
        ],
    },
    packages=find_packages(),
    package_data={
    },
    include_package_data=True,
    license='GPLv3',
)
