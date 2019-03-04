from ase.build import mx2
from gpaw import GPAW, PW, FermiDirac


structure = mx2(formula='WSe2', kind='2H', a=3.319, thickness=3.359,
                size=(1, 1, 1), vacuum=5)
structure.pbc = (1, 1, 1)

ecut = 600

calc = GPAW(mode=PW(ecut),
            xc='LDA',
            kpts={'size': (30, 30, 1), 'gamma': True},
            occupations=FermiDirac(0.01),
            parallel={'domain': 1},
            txt='WSe2_gs_out.txt')

structure.set_calculator(calc)
structure.get_potential_energy()

calc.diagonalize_full_hamiltonian(nbands=500, expert=True)
calc.write('WSe2_gs_fulldiag.gpw', 'all')
