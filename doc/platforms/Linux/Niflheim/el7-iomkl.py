import os
libraries = ['mpi',
             'mkl_intel_lp64',
             'mkl_sequential',
             'mkl_core',
             'pthread',
             'm',
             'dl',
             'readline',
             'xc']
libraries += ['mkl_scalapack_lp64',
              'mkl_blacs_openmpi_lp64']
define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
platform_id = os.environ['CPU_ARCH'] + '-el7'

# Important: libxc must come BEFORE what is already in this list (/usr/lib64)
library_dirs = [os.environ['EBROOTLIBXC']+'/lib'] + library_dirs
include_dirs += [os.environ['EBROOTLIBXC']+'/include']
