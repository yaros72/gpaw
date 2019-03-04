#!/usr/bin/env python
# Copyright (C) 2003-2016  CAMP
# Please see the accompanying LICENSE file for further information.

import distutils
import distutils.util
import os
import os.path as op
import re
import subprocess
import sys
from distutils.command.build_ext import build_ext as _build_ext
from distutils.command.build_scripts import build_scripts as _build_scripts
from distutils.command.sdist import sdist as _sdist
from distutils.core import setup, Extension
from glob import glob

from config import (get_system_config, check_dependencies,
                    write_configuration, build_interpreter, get_config_vars)


assert sys.version_info >= (2, 7)

# Get the current version number:
with open('gpaw/__init__.py', 'rb') as fd:
    txt = fd.read().decode('UTF-8')
version = re.search("__version__ = '(.*)'", txt).group(1)

description = 'GPAW: DFT and beyond within the projector-augmented wave method'
long_description = """\
GPAW is a density-functional theory (DFT) Python code based on the
projector-augmented wave (PAW) method and the atomic simulation environment
(ASE). It uses plane-waves, atom-centered basis-functions or real-space
uniform grids combined with multigrid methods."""

libraries = []
library_dirs = []
include_dirs = []
extra_link_args = []
extra_compile_args = []
runtime_library_dirs = []
extra_objects = []
define_macros = [('NPY_NO_DEPRECATED_API', 7)]
undef_macros = []

mpi_libraries = []
mpi_library_dirs = []
mpi_include_dirs = []
mpi_runtime_library_dirs = []
mpi_define_macros = []

# Search and store current git hash if possible
try:
    from ase.utils import search_current_git_hash
    githash = search_current_git_hash('gpaw')
    if githash is not None:
        define_macros += [('GPAW_GITHASH', githash)]
    else:
        print('.git directory not found. GPAW git hash not written.')
except ImportError:
    print('ASE not found. GPAW git hash not written.')

platform_id = ''

packages = []
for dirname, dirnames, filenames in os.walk('gpaw'):
        if '__init__.py' in filenames:
            packages.append(dirname.replace('/', '.'))

import_numpy = True
if '--ignore-numpy' in sys.argv:
    import_numpy = False
    sys.argv.remove('--ignore-numpy')

remove_default_flags = False
if '--remove-default-flags' in sys.argv:
    remove_default_flags = True
    sys.argv.remove('--remove-default-flags')

customize = 'customize.py'
for i, arg in enumerate(sys.argv):
    if arg.startswith('--customize'):
        customize = sys.argv.pop(i).split('=')[1]
        break

get_system_config(define_macros, undef_macros,
                  include_dirs, libraries, library_dirs,
                  extra_link_args, extra_compile_args,
                  runtime_library_dirs, extra_objects,
                  import_numpy)

error = subprocess.call(['which', 'mpicc'], stdout=subprocess.PIPE)
if error:
    mpicompiler = None
else:
    mpicompiler = 'mpicc'
mpilinker = mpicompiler

compiler = None
scalapack = False
libvdwxc = False
elpa = False

# User provided customizations:
exec(open(customize).read())

if platform_id != '':
    my_platform = distutils.util.get_platform() + '-' + platform_id

    def my_get_platform():
        return my_platform
    distutils.util.get_platform = my_get_platform

if compiler is not None:
    # A hack to change the used compiler and linker:
    vars = get_config_vars()
    if remove_default_flags:
        for key in ['BASECFLAGS', 'CFLAGS', 'OPT', 'PY_CFLAGS',
                    'CCSHARED', 'CFLAGSFORSHARED', 'LINKFORSHARED',
                    'LIBS', 'SHLIBS']:
            if key in vars:
                value = vars[key].split()
                # remove all gcc flags (causing problems with other compilers)
                for v in list(value):
                    value.remove(v)
                vars[key] = ' '.join(value)
    for key in ['CC', 'LDSHARED']:
        if key in vars:
            value = vars[key].split()
            # first argument is the compiler/linker.  Replace with mpicompiler:
            value[0] = compiler
            vars[key] = ' '.join(value)

if scalapack:
    define_macros.append(('GPAW_WITH_SL', '1'))

if libvdwxc:
    define_macros.append(('GPAW_WITH_LIBVDWXC', '1'))

if elpa:
    define_macros.append(('GPAW_WITH_ELPA', '1'))

# distutils clean does not remove the _gpaw.so library and gpaw-python
# binary so do it here:
plat = distutils.util.get_platform()
plat = plat + '-' + sys.version[0:3]
gpawso = 'build/lib.%s/' % plat + '_gpaw.so'
gpawbin = 'build/bin.%s/' % plat + 'gpaw-python'
if 'clean' in sys.argv:
    if op.isfile(gpawso):
        print('removing ', gpawso)
        os.remove(gpawso)
    if op.isfile(gpawbin):
        print('removing ', gpawbin)
        os.remove(gpawbin)

sources = glob('c/*.c') + ['c/bmgs/bmgs.c']
sources = sources + glob('c/xc/*.c')
# Make build process deterministic (for "reproducible build" in debian)
sources.sort()

check_dependencies(sources)

extensions = [Extension('_gpaw',
                        sources,
                        libraries=libraries,
                        library_dirs=library_dirs,
                        include_dirs=include_dirs,
                        define_macros=define_macros,
                        undef_macros=undef_macros,
                        extra_link_args=extra_link_args,
                        extra_compile_args=extra_compile_args,
                        runtime_library_dirs=runtime_library_dirs,
                        extra_objects=extra_objects)]

files = ['gpaw-analyse-basis', 'gpaw-basis',
         'gpaw-mpisim', 'gpaw-plot-parallel-timings', 'gpaw-runscript',
         'gpaw-setup', 'gpaw-upfplot', 'gpaw']
scripts = [op.join('tools', script) for script in files]

write_configuration(define_macros, include_dirs, libraries, library_dirs,
                    extra_link_args, extra_compile_args,
                    runtime_library_dirs, extra_objects, mpicompiler,
                    mpi_libraries, mpi_library_dirs, mpi_include_dirs,
                    mpi_runtime_library_dirs, mpi_define_macros)


class sdist(_sdist):
    """Fix distutils.

    Distutils insists that there should be a README or README.txt,
    but GitLab.com needs README.rst in order to parse it as
    reStructuredText."""

    def warn(self, msg):
        if msg.startswith('standard file not found: should have one of'):
            self.filelist.append('README.rst')
        else:
            _sdist.warn(self, msg)


class build_ext(_build_ext):
    def run(self):
        _build_ext.run(self)
        if mpicompiler:
            # Also build gpaw-python:
            error = build_interpreter(
                define_macros, include_dirs, libraries,
                library_dirs, extra_link_args, extra_compile_args,
                runtime_library_dirs, extra_objects,
                mpicompiler, mpilinker, mpi_libraries,
                mpi_library_dirs,
                mpi_include_dirs,
                mpi_runtime_library_dirs, mpi_define_macros)
            assert error == 0


class build_scripts(_build_scripts):
    # Python 3's version will try to read the first line of gpaw-python
    # because it thinks it is a Python script that need an adjustment of
    # the Python version.  We skip this in our version.
    def copy_scripts(self):
        self.mkpath(self.build_dir)
        outfiles = []
        updated_files = []
        for script in self.scripts:
            outfile = op.join(self.build_dir, op.basename(script))
            outfiles.append(outfile)
            updated_files.append(outfile)
            self.copy_file(script, outfile)
        return outfiles, updated_files


if mpicompiler:
    scripts.append('build/bin.%s/' % plat + 'gpaw-python')

setup(name='gpaw',
      version=version,
      description=description,
      long_description=long_description,
      maintainer='GPAW-community',
      maintainer_email='gpaw-users@listserv.fysik.dtu.dk',
      url='https://wiki.fysik.dtu.dk/gpaw',
      license='GPLv3+',
      platforms=['unix'],
      packages=packages,
      ext_modules=extensions,
      scripts=scripts,
      cmdclass={'sdist': sdist,
                'build_ext': build_ext,
                'build_scripts': build_scripts},
      classifiers=[
          'Development Status :: 6 - Mature',
          'License :: OSI Approved :: '
          'GNU General Public License v3 or later (GPLv3+)',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Physics'])
