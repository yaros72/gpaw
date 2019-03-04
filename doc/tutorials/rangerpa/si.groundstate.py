from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW

bulk_si = bulk('Si', a=5.42935602)
calc = GPAW(mode=PW(400.0),
            xc='LDA',
            occupations=FermiDirac(width=0.01),
            kpts={'size': (4, 4, 4), 'gamma': True},
            parallel={'domain': 1},
            txt='si.gs.txt')

bulk_si.set_calculator(calc)
E_lda = bulk_si.get_potential_energy()
calc.diagonalize_full_hamiltonian()
calc.write('si.lda_wfcs.gpw', mode='all')
