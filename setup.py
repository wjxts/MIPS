#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import os
from setuptools import setup, find_packages, Extension
import sys

if sys.version_info < (3, 8):
    sys.exit("Sorry, Python >= 3.8 is required for vemol.")
    
extensions = [
]
setup(
    name="vemol",
    version="1.0",
    description="Vemol: A Python library for molecular self-supervised learning",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GPL License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    setup_requires=[
        'hydra-core',
        'dgl',
        'torch',
        # 'scikit-learn',
        'dgllife==0.3.1',
    ],
    install_requires=[
    ],
    entry_points={
            "console_scripts": [
                "vemol-train = train:main",
            ],
        },
    packages=find_packages(include=["vemol"]),
    ext_modules=extensions,
)