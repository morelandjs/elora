from setuptools import setup
import sys


def setup_package():
    needs_sphinx = {'docs', 'build_sphinx'}.intersection(sys.argv)

    setup_requires = ['pytest-runner']

    if needs_sphinx:
        setup_requires.append('sphinx>=2.4.3')

    setup(setup_requires=setup_requires)


if __name__ == '__main__':
    setup_package()
