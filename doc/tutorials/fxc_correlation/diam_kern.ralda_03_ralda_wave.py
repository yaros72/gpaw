from gpaw.xc.fxc import FXCCorrelation
from ase.parallel import paropen

fxc = FXCCorrelation('diam_kern.ralda.lda_wfcs.gpw', xc='rALDA',
                     txt='diam_kern.ralda_03_ralda_wave.txt',
                     av_scheme='wavevector')
E_i = fxc.calculate(ecut=[131.072])

resultfile = paropen('diam_kern.ralda_kernel_comparison.dat', 'w')
resultfile.write(str(E_i[-1]) + '\n')
resultfile.close()
