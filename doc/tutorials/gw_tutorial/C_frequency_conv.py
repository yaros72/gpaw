from ase.build import bulk

from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.response.g0w0 import G0W0

a = 3.567
atoms = bulk('C', 'diamond', a=a)

calc = GPAW(mode=PW(300),
            parallel={'domain': 1},
            kpts=(3, 3, 3),
            xc='LDA',
            occupations=FermiDirac(0.001),
            txt='C_groundstate_freq.txt')

atoms.set_calculator(calc)
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian()
calc.write('C_groundstate_freq.gpw', 'all')

for i, domega0 in enumerate([0.01, 0.02, 0.03, 0.04, 0.05]):
    for j, omega2 in enumerate([1, 5, 10, 15, 20, 25]):
        gw = G0W0(calc='C_groundstate_freq.gpw',
                  nbands=30,
                  bands=(3, 5),
                  kpts=[0],
                  ecut=20,
                  domega0=domega0,
                  omega2=omega2,
                  filename='C_g0w0_domega0_%s_omega2_%s' % (domega0, omega2))

        results = gw.calculate()
