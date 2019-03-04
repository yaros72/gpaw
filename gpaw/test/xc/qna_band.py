from __future__ import print_function
from gpaw import GPAW, PW, restart
from ase.lattice.compounds import L1_2

name = 'Cu3Au'
ecut = 300
kpts = (2, 2, 2)

QNA = {'alpha': 2.0,
       'name': 'QNA',
       'orbital_dependent': False,
       'parameters': {'Au': (0.125, 0.1), 'Cu': (0.0795, 0.005)},
       'setup_name': 'PBE',
       'type': 'qna-gga'}

atoms = L1_2(['Au', 'Cu'], latticeconstant=3.74)

calc = GPAW(mode=PW(ecut),
            xc=QNA,
            kpts=kpts,
            parallel={'domain': 1},
            txt='gs.txt')

atoms.set_calculator(calc)
atoms.get_potential_energy()
eigs = calc.get_eigenvalues(kpt=0)[:24]
calc.write('gs.gpw')

atoms, calc = restart('gs.gpw',
                      parallel={'domain': 1},
                      fixdensity=True,
                      kpts=[[-0.25, 0.25, -0.25]])
atoms.get_potential_energy()
eigs_new = calc.get_eigenvalues(kpt=0)[:24]
for eold, enew in zip(eigs, eigs_new):
    print(eold, enew, eold - enew)
    assert abs(eold - enew) < 27e-6
