from setuptools.extension import Extension
from setuptools import setup
from Cython.Build import cythonize
import numpy

ext = Extension('cython_mcmc.mcmc',
                 sources=['cython_mcmc/mcmc.pyx',
                          '../mt19937/srs/mt19937.pyx',
			  '../mt19937/srs/distributions.c',
			  '../mt19937/srs/aligned_malloc.c',
			  '../mt19937/srs/src/random-kit/random-kit.c',
			  '../mt19937/srs/interface/random-kit/random-kit-shim.c'],
                 include_dirs=[numpy.get_include(), '../mt19937/srs'])

setup(name='cython_mcmc', ext_modules=cythonize([ext], include_path=['../mt19937']))
