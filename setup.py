from distutils.core import setup
from setuptools import find_packages
import os
import json

setup(
    name='MOTorNOT',
    version='0.1',
    description='Magneto-optical trap simulation package',
    author='Robert Fasano',
    author_email='robert.j.fasano@colorado.edu',
    packages=find_packages('myproject'),
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=['ipython', 'matplotlib', 'pandas', 'scipy']
)
