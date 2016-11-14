# coding=utf-8

"""
Setup the package 'kaggle-titanic'
"""

from distutils.core import setup

setup(
    name='kaggle-titanic',
    version='0.1',
    packages=['IpynbLoader', 'random_forests', 'python_ii_pandas'],
    url='https://github.com/InonS/kaggle-titanic',
    license='',
    author='Inon Sharony',
    author_email='Inon.Sharony@gmail.com',
    description='https://www.kaggle.com/c/titanic'
)


def configuration():
    """
    NumPy distutils
    """
    from numpy.distutils.misc_util import Configuration
    config = Configuration('kaggle-titanic')
    config.add_subpackage('random_forests')

    return config
