from gpaw.response.g0w0 import G0W0
from ase.build import mx2
from gpaw import GPAW, PW, FermiDirac

structure = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.127,
                size=(1, 1, 1), vacuum=3.5)
structure.pbc = (1, 1, 1)

Ecut = 400

calc = GPAW(mode=PW(Ecut),
            parallel={'domain': 1},
            xc='PBE',
            basis='dzp',
            kpts={'size': (9, 9, 1), 'gamma': True},
            occupations=FermiDirac(0.01),
            txt='MoS2_out_gs.txt')

structure.set_calculator(calc)
structure.get_potential_energy()
calc.write('MoS2_gs.gpw', 'all')

calc.diagonalize_full_hamiltonian()
calc.write('MoS2_fulldiag.gpw', 'all')

for ecut in [80]:
    gw = G0W0(calc='MoS2_fulldiag.gpw',
              bands=(8, 18),
              ecut=ecut,
              truncation='2D',
              nblocksmax=True,
              q0_correction=True,
              filename='MoS2_g0w0_{}'.format(ecut),
              savepckl=True)

    gw.calculate()
