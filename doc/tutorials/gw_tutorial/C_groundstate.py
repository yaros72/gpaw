from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW

a = 3.567
atoms = bulk('C', 'diamond', a=a)

calc = GPAW(mode=PW(300),  # energy cutoff for plane wave basis (in eV)
            kpts={'size': (3, 3, 3), 'gamma': True},
            xc='LDA',
            occupations=FermiDirac(0.001),
            parallel={'domain': 1},
            txt='C_groundstate.txt')

atoms.set_calculator(calc)
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian()  # determine all bands
calc.write('C_groundstate.gpw', 'all')  # write out wavefunctions
