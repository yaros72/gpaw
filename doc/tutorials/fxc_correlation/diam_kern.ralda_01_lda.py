from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW

bulk_c = bulk('C', a=3.5454859)
calc = GPAW(mode=PW(600.0),
            parallel={'domain': 1},
            xc='LDA',
            occupations=FermiDirac(width=0.01),
            kpts={'size': (6, 6, 6), 'gamma': True},
            txt='diam_kern.ralda_01_lda.txt')

bulk_c.set_calculator(calc)
E_lda = bulk_c.get_potential_energy()
calc.diagonalize_full_hamiltonian()
calc.write('diam_kern.ralda.lda_wfcs.gpw', mode='all')
