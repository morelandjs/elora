#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools


def setup_package():
    setuptools.setup(package_data={'elora': ['nfl.dat']})


if __name__=='__main__':
    setup_package()
