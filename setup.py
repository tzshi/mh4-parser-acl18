#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name='acl18',
    version='0.0.1',
    ext_modules=cythonize(
        [
            Extension("acl18.chu_liu_edmonds", ["acl18/chu_liu_edmonds.pyx"],
                   language="c++"),
        ]
    )
)
