compiler='mpicc'
libraries = ['xc']
extra_link_args += [
     '-Wl,--no-as-needed',
    '-lmkl_scalapack_lp64',
    '-lmkl_intel_lp64',
    '-lmkl_core',
    '-lmkl_sequential',
    '-lmkl_blacs_intelmpi_lp64',
]
