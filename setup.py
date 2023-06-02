#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup, find_packages
from distutils.util import convert_path

def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding="utf-8").read()

main_ns = {}
ver_path = convert_path('src/andromeda_chain/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="andromeda-chain",
    version=main_ns["__version__"],
    author="Paolo Rechia",
    author_email="paolorechia@gmail.com",
    maintainer="Paolo Rechia",
    maintainer_email="paolorechia@gmail.com",
    license="MIT",
    url="https://github.com/ChuloAI/andromeda-chain",
    description="A client to query a HTTP Guidance server",
    python_requires=">=3.9",
    install_requires=["requests"],
    package_dir={'':'src'},
    packages=find_packages("src", exclude=["guidance_server"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ]
)