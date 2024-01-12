#!/usr/bin/env python

from distutils.core import setup

setup(
    name='PyTubular',
    version='1.0.0',
    url='https://github.com/juliankappler/PyTubular',
    author='Julian Kappler',
    author_email='jkappler@posteo.de',
    license='GPL3',
    description='Python module for the 1D tubular ensemble',
    packages=['PyTubular',
    'PyTubular.sympy_definitions',
    ],
)
