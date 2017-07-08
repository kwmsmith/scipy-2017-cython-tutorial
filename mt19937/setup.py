import glob
import os
import sys
from os.path import join
import subprocess
import struct

import numpy
from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.dist import Distribution
import versioneer

DEVELOP = False

try:
    import Cython.Tempita as tempita
except ImportError:
    try:
        import tempita
    except ImportError:
        raise ImportError('tempita required to install, '
                          'use pip install tempita')

FORCE_EMULATION = False
USE_SSE2 = True if '--no-sse2' not in sys.argv else False

mod_dir = './srs'
configs = []

rngs = ['RNG_MT19937']

compile_rngs = rngs[:]

extra_defs = [('_CRT_SECURE_NO_WARNINGS', '1')] if os.name == 'nt' else []
extra_link_args = ['/LTCG', 'Advapi32.lib', 'Kernel32.lib'] if os.name == 'nt' else []
base_extra_compile_args = [] if os.name == 'nt' else ['-std=c99']
if USE_SSE2:
    if os.name == 'nt':
        base_extra_compile_args += ['/wd4146', '/GL']
        if struct.calcsize('P') < 8:
            base_extra_compile_args += ['/arch:SSE2']
    else:
        base_extra_compile_args += ['-msse2']


base_include_dirs = [mod_dir] + [numpy.get_include()]
if os.name == 'nt' and sys.version_info < (3, 5):
    base_include_dirs += [join(mod_dir, 'src', 'common')]

for rng in rngs:
    if rng not in compile_rngs:
        continue

    file_name = rng.lower().replace('rng_', '')
    flags = {'RS_RNG_MOD_NAME': file_name}
    sources = [join(mod_dir, file_name + '.pyx'),
               join(mod_dir, 'distributions.c'),
               join(mod_dir, 'aligned_malloc.c')]
    include_dirs = base_include_dirs[:]
    extra_compile_args = base_extra_compile_args[:]

    if rng == 'RNG_MT19937':
        sources += [join(mod_dir, 'src', 'random-kit', p) for p in ('random-kit.c',)]
        sources += [join(mod_dir, 'interface', 'random-kit', 'random-kit-shim.c')]

        defs = [('RS_RANDOMKIT', '1')]

        include_dirs += [join(mod_dir, 'src', 'random-kit')]

    else:
        raise ValueError("what you talkin' bout, Willis?")

    config = {'file_name': file_name,
              'sources': sources,
              'include_dirs': include_dirs,
              'defs': defs,
              'flags': dict([(k, v) for k, v in flags.items()]),
              'compile_args': extra_compile_args
              }

    configs.append(config)


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


try:
    import os
    readme_orig_time = os.path.getmtime('README.md')
    readme_mod_time = os.path.getmtime('README.rst')
    if readme_orig_time > readme_mod_time:
        subprocess.call(['pandoc', '--from=markdown', '--to=rst', '--output=README.rst', 'README.md'])
except:
    pass
# Generate files and extensions
extensions = []

for config in configs:
    config_file_name = mod_dir + '/' + config['file_name'] + '-config.pxi'
    ext = Extension('srs.' + config['file_name'],
                    sources=config['sources'],
                    include_dirs=config['include_dirs'],
                    define_macros=config['defs'] + extra_defs,
                    extra_compile_args=config['compile_args'],
                    extra_link_args=extra_link_args)
    extensions += [ext]

# Do not run cythonize if clean
if 'clean' in sys.argv:
    def cythonize(e, *args, **kwargs):
        return e
else:
    files = glob.glob('./srs/*.in')
    for templated_file in files:
        output_file_name = os.path.splitext(templated_file)[0]
        if (DEVELOP and os.path.exists(output_file_name) and
                (os.path.getmtime(templated_file) < os.path.getmtime(output_file_name))):
            continue
        with open(templated_file, 'r') as source_file:
            template = tempita.Template(source_file.read())
        with open(output_file_name, 'w') as output_file:
            output_file.write(template.substitute())

ext_modules = cythonize(extensions, force=not DEVELOP)

setup(name='srs',  # "simplified randomstate"
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      packages=find_packages(),
      package_dir={'srs': './srs'},
      package_data={'': ['*.c', '*.h', '*.pxi', '*.pyx', '*.pxd'],
                    'srs.tests.data': ['*.csv']},
      include_package_data=True,
      license='NSCA',
      ext_modules=ext_modules)
