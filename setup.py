#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools


def version():
    with open('elora/__init__.py', 'r') as f:
        for l in f:
            if l.startswith('__version__ = '):
                return l.split("'")[1]

    raise RuntimeError('unable to determine version')


def long_description():
    with open('README.rst') as f:
        return f.read()


setuptools.setup(
    name='elora',
    version=version(),
    description='Elo regressor algorithm (elora)',
    long_description=long_description(),
    author='J. Scott Moreland',
    author_email='morelandjs@gmail.com',
    url='https://github.com/elora.git',
    license='MIT',
    packages=['elora'],
    package_data={'elora': ['nfl.dat']},
    install_requires=['numpy', 'scipy >= 0.18.0'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
