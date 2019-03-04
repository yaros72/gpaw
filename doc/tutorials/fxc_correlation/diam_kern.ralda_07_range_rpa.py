from gpaw.xc.fxc import FXCCorrelation
from ase.parallel import paropen

fxc = FXCCorrelation('diam_kern.ralda.lda_wfcs.gpw', xc='range_RPA',
                     txt='diam_kern.ralda_07_range_rpa.txt',
                     range_rc=2.0)
E_i = fxc.calculate(ecut=[131.072, 80.0])

resultfile = paropen('diam_kern.ralda_kernel_comparison.dat', 'a')
resultfile.write(str(E_i[-1]) + '\n')
resultfile.close()
