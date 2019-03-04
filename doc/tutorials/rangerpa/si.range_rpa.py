from gpaw.xc.fxc import FXCCorrelation
from ase.units import Hartree
from ase.parallel import paropen

resultfile = paropen('range_results.dat', 'w')

# Standard RPA result
resultfile.write(str(0.0) + ' ' + str(-12.250) + '\n')

# Suggested parameters from Bruneval, PRL 108, 256403 (2012)
rc_list = [0.5, 1.0, 2.0, 3.0, 4.0]
cutoff_list = [11.0, 5.0, 2.25, 0.75, 0.75]
nbnd_list = [500, 200, 100, 50, 40]

for rc, ec, nbnd in zip(rc_list, cutoff_list, nbnd_list):
    fxc = FXCCorrelation('si.lda_wfcs.gpw',
                         xc='range_RPA',
                         txt='si_range.' + str(rc) + '.txt',
                         range_rc=rc)
    E_i = fxc.calculate(ecut=[ec * Hartree], nbands=nbnd)
    resultfile.write(str(rc) + ' ' + str(E_i[0]) + '\n')
