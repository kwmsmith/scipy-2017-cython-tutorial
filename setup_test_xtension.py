from setuptools import setup
from Cython.Build import cythonize

setup(name='xtension',
      ext_modules=cythonize('xtension/*.pyx'))
