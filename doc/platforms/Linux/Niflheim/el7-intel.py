scalapack = True
platform_id = os.environ['CPU_ARCH'] + '-el7'

# Clean out any autodetected things, we only want the EasyBuild
# definitions to be used.
libraries = []
mpi_libraries = []
include_dirs = []

# Use Intel MKL
libraries += ['fftw3xc_intel','mkl_intel_lp64','mkl_sequential','mkl_core']

# Use EasyBuild scalapack from the active toolchain
mpi_libraries += ['mkl_scalapack_lp64','mkl_blacs_intelmpi_lp64']
mpi_define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
mpi_define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]

# Use EasyBuild libxc
libxc = os.getenv('EBROOTLIBXC')
if libxc:
    include_dirs.append(os.path.join(libxc, 'include'))
    libraries.append('xc')

# libvdwxc:
# Use EasyBuild libvdwxc
# This will only work with the foss toolchain.
libvdwxc = os.getenv('EBROOTLIBVDWXC')
if libvdwxc:
    include_dirs.append(os.path.join(libvdwxc, 'include'))
    libraries.append('vdwxc')

# Now add a EasyBuild "cover-all-bases" library_dirs
library_dirs = os.getenv('LD_LIBRARY_PATH').split(':')

# Build separate gpaw-python
mpicompiler = 'mpiicc'
mpilinker = mpicompiler
