from __future__ import print_function
import pickle
import numpy as np
from gpaw import GPAW, FermiDirac
from gpaw.response.g0w0 import G0W0
from ase.build import bulk
from gpaw.wavefunctions.pw import PW

atoms = bulk('BN', 'zincblende', a=3.615)

calc = GPAW(mode=PW(300),
            parallel={'domain': 1},
            kpts={'size': (3, 3, 3), 'gamma': True},
            xc='LDA',
            occupations=FermiDirac(0.001),
            txt='BN_groundstate.txt')

atoms.set_calculator(calc)
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian()
calc.write('BN_gs_fulldiag.gpw', 'all')

gw = G0W0('BN_gs_fulldiag.gpw',
          bands=(1, 7),
          filename='BN_GW0',
          method='GW0',
          maxiter=5,
          mixing=0.5,
          ecut=50)

gw.calculate()

result = pickle.load(open('BN_GW0_results.pckl', 'rb'))

for i in range(result['iqp'].shape[0]):
    print('Ite:', i,
          'Gap:',
          np.min(result['iqp'][i, 0, :, 3]) -
          np.max(result['iqp'][i, 0, :, 2]))



