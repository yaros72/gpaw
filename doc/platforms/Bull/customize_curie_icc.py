import os

extra_compile_args = ['-std=c99']
compiler = './icc.py'
mpicompiler = './icc.py'
mpilinker = './icc.py'

#libxc
library_dirs = ['/ccc/cont005/home/pawp72/enkovaaj/libxc/lib']
include_dirs += ['/ccc/cont005/home/pawp72/enkovaaj/libxc/include']
libraries = ['z', 'xc']

scalapack = True
hdf5 = True

mkl_flags = os.environ['MKL_SCA_LIBS']
extra_link_args = [mkl_flags]

# HDF5 flags
include_dirs += [os.environ['HDF5_INCDIR']]
libraries += ['hdf5']
library_dirs += [os.environ['HDF5_LIBDIR']]

define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
define_macros += [("GPAW_ASYNC",1)]

