import pickle

import numpy as np
from ase.build import bulk

from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.response.g0w0 import G0W0


a = 3.567
atoms = bulk('C', 'diamond', a=a)

k = 8

calc = GPAW(mode=PW(600, force_complex_dtype=True),
            parallel={'domain': 1},
            kpts={'size': (k, k, k), 'gamma': True},
            dtype=complex,
            xc='LDA',
            occupations=FermiDirac(0.001),
            txt='C_converged_ppa.txt')

atoms.set_calculator(calc)
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian()
calc.write('C_converged_ppa.gpw', 'all')

for ecut in [300, 400]:
    gw = G0W0(calc='C_converged_ppa.gpw',
              kpts=[0],
              bands=(3, 5),
              ecut=ecut,
              ppa=True,
              filename='C-g0w0_ppa_{}'.format(ecut))

    gw.calculate()

fil = pickle.load(open('C-g0w0_ppa_300_results.pckl', 'rb'))
direct_gap_300 = fil['qp'][0, 0, 1] - fil['qp'][0, 0, 0]
fil = pickle.load(open('C-g0w0_ppa_400_results.pckl', 'rb'))
direct_gap_400 = fil['qp'][0, 0, 1] - fil['qp'][0, 0, 0]

extrap_gap, slope = np.linalg.solve(np.array([[1, 1. / 300.**(3. / 2)],
                                              [1, 1. / 400.**(3. / 2)]]),
                                    np.array([direct_gap_300, direct_gap_400]))
print('Direct gap:', extrap_gap)
