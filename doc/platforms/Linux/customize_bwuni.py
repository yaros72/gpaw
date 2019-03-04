compiler='mpicc'
extra_link_args += [
     '-Wl,--no-as-needed',
    '-L/opt/bwhpc/common/compiler/intel/compxe.2013.sp1.4.211/mkl/lib/intel64',
    '-lmkl_scalapack_lp64',
    '-lmkl_intel_lp64',
    '-lmkl_core',
    '-lmkl_sequential',
    '-lmkl_blacs_intelmpi_lp64',
]
