from setuptools.extension import Extension
from setuptools import setup
from Cython.Build import cythonize
import numpy

ext = Extension('cython_mcmc.mcmc',
                 sources=['cython_mcmc/mcmc.pyx'],
                 include_dirs=[numpy.get_include(), '../rng/rng'])

setup(name='cython_mcmc', ext_modules=cythonize([ext], include_path=['../rng']))
