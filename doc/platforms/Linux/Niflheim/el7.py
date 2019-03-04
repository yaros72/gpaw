import os
scalapack = True
libraries = ['readline',
             'gfortran',
             'scalapack',
             'openblas',
             'xc']
extra_compile_args = ['-O2', '-std=c99', '-fPIC', '-Wall',
                      '-Wno-unknown-pragmas']
define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1'),
                  ('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
platform_id = os.environ['CPU_ARCH'] + '-el7'

# Important: libxc must come BEFORE what is already in this list (/usr/lib64)
library_dirs = [os.environ['EBROOTLIBXC'] + '/lib'] + library_dirs
include_dirs += [os.environ['EBROOTLIBXC'] + '/include']
