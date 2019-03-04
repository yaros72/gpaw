"""GBAR customize.py for GPAW."""

from pathlib import Path

xc = str(Path.home()) + '/'  # libxc folder
libraries = ['xc', 'openblas', 'scalapack']
library_dirs = [
    xc + 'lib',
    '/appl/ScaLAPACK/2.0.2-openblas-0.2.15//XeonE5-2670v3/gcc-6.1.0/lib',
    '/appl/OpenBLAS/0.2.14/XeonE5-2670v3/gcc/lib']
scalapack = True
define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
include_dirs += [xc + 'include']
library_dirs += [xc + 'lib']
extra_link_args += ['-Wl,-rpath={xc}/lib'.format(xc=xc)]
