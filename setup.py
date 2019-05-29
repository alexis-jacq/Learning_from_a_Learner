#!/usr/bin/env python
 # -*- coding: utf-8 -*-

from setuptools import find_packages, setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
  name='lfl',
  version='0.0.1',
  license='Apache License',
  description='Implements code from LfL paper (http://proceedings.mlr.press/v97/jacq19a/jacq19a.pdf).',
  long_description=readme(),
  classifiers=[
    'License :: OSI Approved :: Apache License',
    'Programming Language :: Python :: 2.7',
  ],
  author='Alexis David Jacq',
  author_email='alexis.jacq@gmail.com',
  url='https://github.com/alexis-jacq/Learning_from_a_Learner',
  packages=find_packages(),
  )
